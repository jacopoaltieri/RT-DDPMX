import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, all_reduce
import os


from dataset import PNGDataset
from models import UNet
import utils
from tqdm import tqdm
import datetime

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 scaler,
                 save_every: int,
                 snapshot_path: str) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

        self.losses=[]
    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def noise_estimation_loss(self, model, x0: torch.Tensor, t: torch.LongTensor, e: torch.Tensor, b: torch.Tensor, keepdim=False):
        # Compute alpha_t for the given timesteps t
        a = torch.cumprod(1 - b, dim=0).index_select(0, t).view(-1, 1, 1, 1).to(x0.device)
        # Generate the noisy images at timestep t
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        
        # Forward pass through the model
        output = model(x, t)
        
        # Compute the loss (per-sample if keepdim=True)
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    def _run_batch(self, images, timesteps):
        self.optimizer.zero_grad()
        images = images.to(self.gpu_id)
        timesteps = timesteps.to(self.gpu_id)

        # Get beta schedule and compute noise
        beta_schedule = utils.get_beta_schedule('linear', beta_start=0.0001, beta_end=0.006, num_diffusion_timesteps=100)
        beta = torch.tensor(beta_schedule, dtype=torch.float32).to(images.device)
        
        # Sample Gaussian noise
        noise = torch.randn_like(images).to(images.device)
        timesteps = timesteps.to(images.device)

        # Compute noise estimation loss using the corrected function
        with autocast(device_type='cuda'):
            loss = self.noise_estimation_loss(self.model, images, timesteps, noise, beta)

        # Backward pass and optimization with mixed precision
        self.scaler.scale(loss).backward()
        # All-reduce to synchronize gradients across all GPUs
        self.scaler.unscale_(self.optimizer)  # Unscale gradients before all_reduce
        for param in self.model.parameters():
            if param.grad is not None:
                all_reduce(param.grad.data)  # Synchronize gradients

        # Step the optimizer and update with mixed precision
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Step the scheduler after the optimizer step
        self.scheduler.step()

        return loss.item()

    def _run_epoch(self, epoch):
        if self.gpu_id == 0:
            progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch} [GPU{self.gpu_id}]", leave=False)
        else:
            progress_bar = self.train_data

        total_loss = 0
        num_batches = 0

        for source, targets in progress_bar:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss = self._run_batch(source, targets)
            total_loss += batch_loss
            num_batches += 1

            if self.gpu_id == 0:
                progress_bar.set_postfix(loss=batch_loss)

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)

        if self.gpu_id == 0:
            print(f"[{datetime.datetime.now()}] [GPU{self.gpu_id}] Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        return avg_loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs+1):
            avg_loss = self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        
        # Plot the training loss
        plt.figure(figsize=(10, 10))
        plt.plot(range(len(self.losses)), self.losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig('training_loss_plot.png')  # Save the plot as an image
        plt.close()
        
def load_train_objs(config):
    dataset = PNGDataset(config["dataset"]["directory"])
    model = UNet(
        in_ch=config["model"]["in_ch"],
        out_ch=config["model"]["out_ch"],
        resolution=config["model"]["resolution"],
        num_res_blocks=config["model"]["num_res_blocks"],
        ch=config["model"]["ch"],
        ch_mult=tuple(config["model"]["ch_mult"]),
        attn_resolutions=config["model"]["attn_resolutions"],
        dropout=config["model"]["dropout"],
        resamp_with_conv=config["model"]["resamp_with_conv"],
    )
    epochs = config["training"]["num_epochs"]
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scaler = GradScaler()
    return dataset, model, optimizer, scheduler, scaler

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(config):
    batch_size = config["training"]["batch_size"]
    total_epochs = config["training"]["num_epochs"]
    save_every = config["training"]["save_every"]
    snapshot_path = config["training"]["snapshot_path"]

    dataset, model, optimizer, scheduler, scaler = load_train_objs(config)

    # Set up DDP for distributed training
    utils.ddp_setup()
    train_data = prepare_dataloader(dataset, batch_size)

    # Create a Trainer instance and start training
    trainer = Trainer(model, train_data, optimizer, scheduler, scaler, save_every, snapshot_path)
    trainer.train(total_epochs)

    # Destroy the process group after training
    destroy_process_group()

if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
