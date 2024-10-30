import os
import torch
from models import UNet
import numpy as np
import utils
from tqdm import tqdm
from collections import defaultdict


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
    estimated_timestep = max(1, min(estimated_timestep.item(), len(betas) - 1))  # Ensure within valid range
    return estimated_timestep


def estimate_average_timestep_for_image(rois, betas, noise_type="gaussian"):
    """
    Estimate the average timestep based on the noise variance of all ROIs of the same image.

    Parameters:
    - rois (list of torch.Tensor): List of tensors, each representing a noisy ROI.
    - betas (torch.Tensor): Beta values (variance schedule) used during diffusion training.
    - noise_type (str): Type of noise ("gaussian" or "poisson").

    Returns:
    - int: Average estimated diffusion timestep for the image.
    """
    timesteps = []
    for roi in rois:
        estimated_timestep = estimate_timestep(roi, betas, noise_type)
        timesteps.append(estimated_timestep)
    
    # Calculate average timestep and round to nearest integer
    average_timestep = int(round(np.mean(timesteps)))
    return average_timestep


def denoise_image(model, noisy_image, beta_schedule, starting_timestep):
    x_t = noisy_image
    for t in reversed(range(1, starting_timestep + 1)):
        with torch.no_grad():
            eta_theta = model(x_t, torch.tensor([t], device=x_t.device))
        
        beta_t = beta_schedule[t]
        beta_t = torch.tensor(beta_t, device=x_t.device) if not isinstance(beta_t, torch.Tensor) else beta_t
        x_t = (1 / torch.sqrt(1 - beta_t)) * (x_t - beta_t * eta_theta)
    return x_t.cpu()


def process_images_in_folder_with_avg_timestep(config, model, betas, input_folder: str, output_folder: str, device):
    """
    Process all images in a specified folder, averaging timesteps for ROIs of the same image.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to group ROIs by base image name
    image_groups = defaultdict(list)

    # Group ROIs by base image name
    for image_file in sorted(os.listdir(input_folder)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = "_".join(image_file.split("_")[:2])  # Adjust based on ROI naming format
            image_path = os.path.join(input_folder, image_file)
            image_groups[base_name].append(image_path)

    # Process each group of ROIs
    for base_name, image_paths in tqdm(image_groups.items(), desc="Processing image groups"):
        # Load all ROIs for the image
        rois = [utils.load_image_as_tensor(image_path, device=device) for image_path in image_paths]
        betas_tensor = torch.from_numpy(betas).float().to(device)

        # Estimate the average timestep for the ROIs of this image
        avg_timestep = estimate_average_timestep_for_image(rois, betas_tensor, "gaussian")
        print(f"Average timestep for {base_name}: {avg_timestep}")

        # Denoise each ROI using the averaged timestep
        for image_path in image_paths:
            test_image = utils.load_image_as_tensor(image_path, device=device)
            denoised_image = denoise_image(model, test_image, betas, avg_timestep)

            # Save the denoised image
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_denoised.png")
            utils.save_image(denoised_image, output_file_path)
            print(f"Denoised image saved as: {output_file_path}")


def main(config):
    input_path = "/home/jaltieri/ddpmx/rois"
    output_folder = "/home/jaltieri/ddpmx/output"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model from snapshot
    model = utils.load_model(config, config["training"]["snapshot_path"], device=device)
    beta_schedule = utils.get_beta_schedule('linear', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=100)

    # Process all ROIs in the input folder
    process_images_in_folder_with_avg_timestep(config, model, beta_schedule, input_path, output_folder, device)


if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
