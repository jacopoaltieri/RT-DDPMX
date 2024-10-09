import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import PNGDataset
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.amp import autocast, GradScaler
from models import UNet
import utils
import losses
from tqdm import tqdm
import logging

def setup_logging(log_file='training.log'):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def setup_ddp(rank, world_size):
    if rank == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def create_dataloaders(image_dir, batch_size=2, train_ratio=0.7, val_ratio=0.15, rank=0, world_size=1):
    dataset = PNGDataset(image_dir)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Use DistributedSampler for training data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(rank, world_size, model, dataloader, num_epochs=100, learning_rate=1e-4):
    model.train()
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        logging.info(f"Rank {rank}: Starting epoch {epoch + 1}/{num_epochs}...")

        for batch_idx, (images, timesteps) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=(rank != 0))
        ):
            images = images.to(device)
            timesteps = timesteps.to(device)
            noise = torch.randn_like(images)
            noisy_images = images + noise

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                predicted_noise = model(noisy_images, timesteps)
                loss = losses.mse_loss(predicted_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / (batch_idx + 1)
        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")

    logging.info(f"Rank {rank}: Training complete!")

def main(rank, world_size):
    # Set up logging at the beginning of the main function
    setup_logging(log_file='ddpmx_training.log')

    # Set up DDP
    setup_ddp(rank, world_size)

    # Load configuration and model parameters
    config = utils.load_yaml("cfg.yaml")
    in_ch = config["model"]["in_ch"]
    out_ch = config["model"]["out_ch"]
    resolution = config["model"]["resolution"]
    num_res_blocks = config["model"]["num_res_blocks"]
    ch = config["model"]["ch"]
    ch_mult = tuple(config["model"]["ch_mult"])
    attn_resolutions = config["model"]["attn_resolutions"]
    dropout = config["model"]["dropout"]
    resamp_with_conv = config["model"]["resamp_with_conv"]

    # Initialize the model
    model = UNet(
        in_ch=in_ch,
        out_ch=out_ch,
        resolution=resolution,
        num_res_blocks=num_res_blocks,
        ch=ch,
        ch_mult=ch_mult,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        resamp_with_conv=resamp_with_conv,
    )

    # Create DataLoaders
    image_directory = config["dataset"]["directory"]
    batch_size = config["training"]["batch_size"]

    logging.info("Creating dataloaders")
    train_loader, val_loader, test_loader = create_dataloaders(image_directory, batch_size, rank=rank, world_size=world_size)

    # Train the model
    logging.info("Starting training")
    train_model(
        rank,
        world_size,
        model,
        train_loader,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
    )

    # Clean up DDP
    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
