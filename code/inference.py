import os
import torch
from models import UNet
import numpy as np
import utils
from tqdm import tqdm  # Import tqdm for the progress bar


def estimate_timestep(image: torch.Tensor, betas: torch.Tensor, noise_type="gaussian"):
    """
    Estimate the timestep based on the noise variance of a noisy image and the beta schedule.

    Parameters:
    - image (torch.Tensor): Noisy image tensor.
    - betas (torch.Tensor): Beta values (variance schedule) used during diffusion training.
    - noise_type (str): Type of noise ("gaussian" or "poisson").
    
    Returns:
    - int: Estimated diffusion timestep.
    """
    
    # Calculate cumulative product of (1 - beta) over timesteps (alpha terms)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # Normalize image to [0, 1] intensity range for variance estimation
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Calculate the variance of the noisy image
    noise_variance = image_norm.var().item()

    if noise_type.lower() == "gaussian":
        # Expected variance for Gaussian noise at each timestep
        expected_variances = 1 - alpha_cumprod

    elif noise_type.lower() == "poisson":
        # Adjusted expected variances for Poisson noise
        expected_variances = (1 - alpha_cumprod) * noise_variance

    else:
        raise ValueError("noise_type must be 'gaussian' or 'poisson'")

    # Find the closest expected variance to the observed image variance
    estimated_timestep = torch.abs(expected_variances - noise_variance).argmin()

    # Ensure the timestep is within the valid range [1, len(betas) - 1]
    estimated_timestep = max(1, min(estimated_timestep.item(), len(betas) - 1))
    
    return estimated_timestep


def denoise_image(model, noisy_image, beta_schedule, starting_timestep):
    x_t = noisy_image
    
    for t in reversed(range(1, starting_timestep + 1)):
        with torch.no_grad():
            eta_theta = model(x_t, torch.tensor([t], device=x_t.device))
        
        beta_t = beta_schedule[t]
        beta_t = torch.tensor(beta_t, device=x_t.device) if not isinstance(beta_t, torch.Tensor) else beta_t

        x_t = (1 / torch.sqrt(1 - beta_t)) * (x_t - beta_t * eta_theta)
        
    return x_t.cpu()


def process_image(config, image_path: str, output_folder: str, model, beta_schedule, device):
    """
    Process a single image for denoising.
    """
    # Load the image as a tensor
    test_image = utils.load_image_as_tensor(image_path, device=device)
    betas = torch.from_numpy(beta_schedule).float().to(device)

    # Estimate the appropriate timestep based on image variance
    timestep = estimate_timestep(test_image, betas, "gaussian")
    print(f"Estimated timestep for {image_path}: {timestep}")

    # Denoise the image starting from the estimated timestep
    denoised_image = denoise_image(model, test_image, beta_schedule, timestep)

    # Save the denoised image to disk with the original name plus a suffix
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_denoised.png")
    utils.save_image(denoised_image, output_file_path)
    print(f"Denoised image saved as: {output_file_path}")


def process_images_in_folder(config, model, betas, input_folder: str, output_folder: str, device):
    """
    Process all images in a specified folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        process_image(config, image_path, output_folder, model, betas, device)


def main(config):
    input_path = "/home/jaltieri/ddpmx/dataset128/R_9598e90d-96cc-49a1-b8cf-51bc2c4d6517_30_unfiltered_frame_23.png"  # Change this to your input folder path
    output_folder = "/home/jaltieri/ddpmx/"  # Output folder name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model from snapshot
    model = utils.load_model(config, config["training"]["snapshot_path"], device=device)
    beta_schedule = utils.get_beta_schedule('linear', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=100)

    if os.path.isfile(input_path):  # Check if the input is a single image
        process_image(config, input_path, output_folder, model, beta_schedule, device)
    elif os.path.isdir(input_path):  # Check if the input is a directory
        process_images_in_folder(config, model, beta_schedule, input_path, output_folder, device)
    else:
        print(f"Input path '{input_path}' is neither a valid file nor a directory.")


if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
