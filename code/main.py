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

# VARIABLES
model_train = True


# FUNCTIONS
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
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

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

            # Reset gradients
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
    
    torch.save(model.state_dict(), "unet_denoiser.pth")
    print("Model saved as unet_denoiser.pth")

def load_model(model_path, device="cuda"):
    # Initialize the model structure (same as training)
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

    # Load the saved weights into the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded from", model_path)
    return model

def denoise_image(model, image, timestep, device="cuda"):
    # Set the model to evaluation mode
    model.eval()

    # Move the image to the appropriate device
    image = image.to(device)
    timestep = torch.tensor([timestep], device=device)

    # Generate Gaussian noise and add it to the image
    noise = torch.randn_like(image)
    noisy_image = image + noise

    # Forward pass to predict the noise with the model
    with torch.no_grad():
        with autocast(device_type=device.type):
            predicted_noise = model(noisy_image.unsqueeze(0), timestep)

    # Subtract the predicted noise from the noisy image to get the denoised image
    denoised_image = noisy_image - predicted_noise.squeeze(0)

    return denoised_image.cpu()



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
    model = model.to(device)

    # Dataset directory from config
    image_directory = config["dataset"]["directory"]
    batch_size = config["training"]["batch_size"]


    if model_train:
        # Create the dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(image_directory, batch_size)
        # Train the model
        train_model(
            model,
            train_loader,  
            num_epochs=config["training"]["num_epochs"],
            learning_rate=config["training"]["learning_rate"],
            device=device,
        )

        # Save the model after training
        torch.save(model.state_dict(), "unet_denoiser.pth")
        print("Model saved as unet_denoiser.pth")

    # Load the model from the saved state for inference
    model = load_model("unet_denoiser.pth", device=device)

    # Assuming you have a single image to test on, load the image as a tensor
    test_image = utils.load_image_as_tensor("/home/jaltieri/ddpmx/dataset128/R_b344347d-2db8-432c-8ca7-f660af506286_160_unfiltered_frame_23.png", device=device)

    # Perform denoising on the test image
    timestep = 10  # Example timestep value
    denoised_image = denoise_image(model, test_image, timestep, device=device)

    # Save or display the output denoised image
    utils.save_image(denoised_image, "denoised_output.png")
    print("Denoised image saved as denoised_output.png")
