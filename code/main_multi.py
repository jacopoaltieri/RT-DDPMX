import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, all_reduce


from dataset import PNGDataset
from models import UNet
import utils


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 test_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 scaler,
                 save_every: int,
                 snapshot_path: str,
                 patience: int) -> None:
        
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.save_every = save_every
        self.patience = patience
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.epochs_since_improvement = 0
        
        self.losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def noise_estimation_loss(self, model, x0: torch.Tensor, t: torch.LongTensor, e: torch.Tensor, b: torch.Tensor):
        """
        Compute the MSE loss between real noise (e) and predicted noise by the model.

        Parameters:
        - model: The U-Net model used for noise prediction.
        - x0: The original image tensor.
        - t: The timesteps at which noise is applied.
        - e: The real Gaussian noise applied to the image.
        - b: The beta schedule used for diffusion.

        Returns:
        - The noise estimation loss (MSE between real and predicted noise).
        """

        # Compute alpha_t for the given timesteps t
        a = torch.cumprod(1 - b, dim=0).index_select(0, t).view(-1, 1, 1, 1).to(x0.device)

        # Generate the noisy images at timestep t
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

        # Forward pass through the model to predict the noise at timestep t
        noise_pred = model(x, t)

        # Compute the MSE loss between the real noise 'e' and the predicted noise 'noise_pred'
        return (e - noise_pred).square().mean(dim=(1, 2, 3)).mean(dim=0)


    def _run_batch(self, images, timesteps, training=True):
        """
        Run a batch for training or validation.
        Parameters:
            - images: The input images
            - timesteps: The diffusion timesteps
            - training: Whether this batch is for training (if False, runs in validation mode)
        """
        if training:
            self.optimizer.zero_grad()

        images = images.to(self.gpu_id)
        timesteps = timesteps.to(self.gpu_id)

        beta_schedule = utils.get_beta_schedule('linear', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=100)
        beta = torch.tensor(beta_schedule, dtype=torch.float32).to(images.device)
        
        noise = torch.randn_like(images).to(images.device)

        with autocast(device_type='cuda'):
            loss = self.noise_estimation_loss(self.model, images, timesteps, noise, beta)

        # In validation, we simply return the loss without backpropagation
        if not training:
            return loss.item()

        # Training phase
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Synchronize gradients for DDP and perform optimization step
        for param in self.model.parameters():
            if param.grad is not None:
                all_reduce(param.grad.data)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    def _run_epoch(self, epoch):
        # Training mode
        self.model.train()
        if self.gpu_id == 0:
            progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch} [GPU{self.gpu_id}]", leave=False)
        else:
            progress_bar = self.train_data

        total_loss = 0
        num_batches = 0

        for source, targets in progress_bar:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss = self._run_batch(source, targets, training=True)
            total_loss += batch_loss
            num_batches += 1

            if self.gpu_id == 0:
                progress_bar.set_postfix(loss=batch_loss)

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)

        if self.gpu_id == 0:
            print(f"[{datetime.datetime.now()}] [GPU{self.gpu_id}] Epoch {epoch} completed. Average Training Loss: {avg_loss:.4f}")

        return avg_loss

    def _run_validation(self):
        # Validation mode
        self.model.eval()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():  # Ensure no gradients are calculated in validation
            for source, targets in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                batch_loss = self._run_batch(source, targets, training=False)
                total_val_loss += batch_loss
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        if self.gpu_id == 0:
            print(f"[{datetime.datetime.now()}] [GPU{self.gpu_id}] Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def _run_testing(self):
        self.model.eval()
        total_test_loss = 0
        num_batches = 0

        with torch.no_grad():
            for source, targets in self.test_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                batch_loss = self._run_batch(source, targets, training=False)
                total_test_loss += batch_loss
                num_batches += 1

        avg_test_loss = total_test_loss / num_batches
        
        if self.gpu_id == 0:
            print(f"[{datetime.datetime.now()}] [GPU{self.gpu_id}] Testing Loss: {avg_test_loss:.4f}")

        return avg_test_loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs + 1):
            avg_loss = self._run_epoch(epoch)
            avg_val_loss = self._run_validation()
            
            # Early stopping logic only if patience is set
            if self.patience  > 0:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.epochs_since_improvement = 0
                    if self.gpu_id == 0:
                        self._save_snapshot(epoch)  # Save snapshot if validation loss improved
                else:
                    self.epochs_since_improvement += 1
                    if self.epochs_since_improvement >= self.patience:
                        print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                        break
            # Optional: Handle case when patience is not used
            else:
                if self.gpu_id == 0:
                    self._save_snapshot(epoch)  # Save snapshot every epoch if not using patience
                
        plt.figure(figsize=(10, 10))
        plt.plot(range(len(self.losses)), self.losses, label='Training Loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig('training_loss_plot.png')
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

def split_dataset(dataset, train_ratio, val_ratio):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

def main(config):
    batch_size = config["training"]["batch_size"]
    total_epochs = config["training"]["num_epochs"]
    save_every = config["training"]["save_every"]
    snapshot_path = config["training"]["snapshot_path"]
    early_stopping_patience = config["training"]["early_stopping_patience"]

    dataset, model, optimizer, scheduler, scaler = load_train_objs(config)
    train_ratio = config["dataset"]["train_ratio"]
    val_ratio = config["dataset"]["val_ratio"]

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio, val_ratio)

    utils.ddp_setup()
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)

    trainer = Trainer(model, train_data, val_data, test_data, optimizer, scheduler, scaler, save_every, snapshot_path, early_stopping_patience)
    trainer.train(total_epochs)
    trainer._run_testing()

    destroy_process_group()

if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
