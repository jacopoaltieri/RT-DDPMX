import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import PNGDataset
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from models import UNet
import utils
import losses
from tqdm import tqdm


def create_dataloaders(image_dir, batch_size=2, train_ratio=0.7, val_ratio=0.15):
    # Create dataset
    dataset = PNGDataset(image_dir)

    # Calculate sizes for splits
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, dataloader, num_epochs=100, learning_rate=1e-4, device="cuda"):
    # Set the model to training mode
    model.train()

    # Define the AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Cosine learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    # Mixed precision training setup
    scaler = GradScaler()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        # Wrap the dataloader with tqdm for progress tracking
        for batch_idx, (images, timesteps) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        ):
            # Move inputs to device
            images = images.to(device)
            timesteps = timesteps.to(device)

            # Generate the Gaussian noise η and add it to the image
            noise = torch.randn_like(images)
            noisy_images = images + noise

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type=device.type):
                predicted_noise = model(noisy_images, timesteps)
                loss = losses.mse_loss(predicted_noise, noise)

            # Backward pass and optimization with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Print the average loss for this epoch
        avg_loss = epoch_loss / (batch_idx + 1)
        print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")

    print("Training complete!")



# Main entry point
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Load parameters from the YAML file
    config = utils.load_yaml("cfg.yaml")

    # Model parameters from config
    in_ch = config["model"]["in_ch"]
    out_ch = config["model"]["out_ch"]
    resolution = config["model"]["resolution"]
    num_res_blocks = config["model"]["num_res_blocks"]
    ch = config["model"]["ch"]
    ch_mult = tuple(config["model"]["ch_mult"])
    attn_resolutions = config["model"]["attn_resolutions"]
    dropout = config["model"]["dropout"]
    resamp_with_conv = config["model"]["resamp_with_conv"]

    # Initialize the UNet model
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
    # Move the model to the GPU if available
    model = model.to(device)

    # Dataset directory from config
    image_directory = config["dataset"]["directory"]  # Assuming your YAML has a 'directory' key under 'dataset'
    batch_size = config["training"]["batch_size"]  # Get batch size from config

    # Create the dataloaders
    # train_loader = utils.generate_mock_dataloader()
    train_loader, val_loader, test_loader = create_dataloaders(image_directory, batch_size)
    
    # Train the model (use train_loader)
    train_model(
        model,
        train_loader,  
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
    )