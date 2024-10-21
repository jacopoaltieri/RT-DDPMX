import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from dataset import PNGDataset  # Assume this is your custom dataset
from models import UNet  # Assume this is your UNet model class
import utils  # For loading configuration

def ddp_setup():
    """
    Set up the Distributed Data Parallel (DDP) environment.
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, save_every: int, snapshot_path: str) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, images, timesteps):
        self.optimizer.zero_grad()
        images = images.to(self.gpu_id)
        timesteps = timesteps.to(self.gpu_id)

        noise_level = 0.1
        noise = noise_level * torch.randn_like(images, device=images.device)
        noisy_images = images + noise

        predicted_noise = self.model(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch size: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for images, timesteps in self.train_data:
            batch_loss = self._run_batch(images, timesteps)
            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            avg_loss = self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

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
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(config):
    ddp_setup()
    
    # Load training parameters from the config file
    batch_size = config["training"]["batch_size"]
    total_epochs = config["training"]["num_epochs"]
    save_every = config["training"]["save_every"]
    snapshot_path = config["training"]["snapshot_path"]

    dataset, model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    
    destroy_process_group()

if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
