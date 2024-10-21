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
import matplotlib.pyplot as plt

# VARIABLES
model_train = True  # Set to True to train the model


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


def train_model(model, dataloader, num_epochs=100, learning_rate=1e-4, device="cuda", patience=5):
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    scaler = GradScaler()

    # Lists to store loss values for plotting
    train_losses = []
    best_loss = float('inf')
    epochs_without_improvement = 0

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
            noise_level = 0.1  # Adjust this value as needed
            noise = noise_level * torch.randn_like(images)
            noisy_images = images + noise


            # Reset gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                predicted_noise = model(noisy_images, timesteps)
                loss = losses.mse_loss(predicted_noise, noise)

            # Backward pass and optimization with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / (batch_idx + 1)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Save the model when it improves
            torch.save(model.state_dict(), "unet_denoiser_best.pth")
            print("Model improved and saved as unet_denoiser.pth")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs.")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete!")
    
    # Plot the training loss
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('training_loss_plot.png')  # Save the plot as an image
    plt.show()  # Show the plot

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


def estimate_gaussian_timestep(image, max_timesteps=1000):
    """
    Estimate the timestep for Gaussian noise based on the noise variance of the image.

    Parameters:
    - image: The noisy image tensor.
    - max_timesteps: The maximum number of timesteps in the diffusion process.

    Returns:
    - timestep: The estimated timestep t' for Gaussian noise denoising.
    """
    # Calculate the noise variance of the image
    noise_variance = image.var().item()
    
    # Define a threshold for minimum variance to avoid division by zero issues
    threshold = 1e-6

    # If variance is below the threshold, set a minimum timestep to avoid underflow
    if noise_variance < threshold:
        return 1  # Start from a minimum timestep instead of 0

    # Compute alpha_t based on the noise variance in relation to its maximum possible value
    alpha_t = min(1.0, noise_variance / (noise_variance + threshold))

    # Estimate the timestep t' using a scaled value that fits the range [0, max_timesteps]
    timestep = int(max_timesteps * (1 - alpha_t))  # Scales alpha_t to map to the timestep range

    # Ensure the timestep is within the allowable range [1, max_timesteps]
    timestep = max(1, min(timestep, max_timesteps))
    return timestep



def denoise(model, noisy_image, timestep, device="cuda"):
    # Set the model to evaluation mode
    model.eval()
    
    # Move the image to the appropriate device
    current_image = noisy_image.to(device)
    
    # Ensure current_image has 4 dimensions (batch_size, channels, height, width)
    if current_image.dim() == 3:  
        current_image = current_image.unsqueeze(0)

    # Convert the timestep to a tensor and move it to the appropriate device
    timestep_tensor = torch.full((1,), timestep, dtype=torch.int, device=device)

    with torch.no_grad():
        for t in range(timestep, -1, -1):
            print(t)
            # Convert the current timestep to a tensor for this iteration
            timestep_tensor.fill_(t)  # Update the tensor value for each step

            # Predict the noise component using the U-Net model, passing the timestep as a tensor
            noise_pred = model(current_image, timestep_tensor)

            # Compute the mean (mu_theta) for the reverse step
            mu_theta = current_image - noise_pred

            # Update the current image based on the reverse step
            current_image = mu_theta + torch.randn_like(current_image) * torch.sqrt(torch.tensor(1 - (t / timestep)))

            # Clip the intensities to the range [0, 1] as described in the paper
            current_image = torch.clamp(current_image, 0.0, 1.0)

    return current_image.cpu()





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
    max_timesteps = config["model"]["max_timesteps"]

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

    # Load the model from the saved state for inference
    model = load_model("unet_denoiser_best.pth", device=device)
    # Assuming you have a single image to test on, load the image as a tensor
    test_image = utils.load_image_as_tensor("/home/jaltieri/ddpmx/dataset128/R_b344347d-2db8-432c-8ca7-f660af506286_160_unfiltered_frame_23.png", device=device)
    # Perform denoising on the test image
    timestep = estimate_gaussian_timestep(test_image)
    denoised_image = denoise(model, test_image, timestep, device=device)

    # Save or display the output denoised image
    utils.save_image(denoised_image, "denoised_output.png")
    print("Denoised image saved as denoised_output.png")
