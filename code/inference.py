import torch
from models import UNet
import numpy as np
import utils

def load_model(config, model_path: str, device: str) -> torch.nn.Module:
    """
    Load a pre-trained model from a .pt file.

    Parameters:
    - config: Configuration dictionary for the model.
    - model_path: The path to the model file (.pt).
    - device: The device to load the model onto (e.g., "cuda" or "cpu").

    Returns:
    - model: The loaded model.
    """
    # Create the model using parameters from config
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

    # Load model weights
    state_dict = torch.load(model_path, map_location=device)
    if 'MODEL_STATE' in state_dict:
        model.load_state_dict(state_dict['MODEL_STATE'])
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def estimate_gaussian_timestep(image:torch.Tensor, betas:torch.Tensor):
    """
    Estimate the timestep based on the variance of the noisy image and the beta schedule.
    
    Parameters:
    - image: Noisy image tensor.
    - betas: Beta values scheduled during training.

    Returns:
    - timestep: Estimated diffusion timestep.
    """
    
    
    alphas = 1- betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    image_norm = (image - image.min()) / (image.max() - image.min())

    # Calculate the variance of the noisy image
    noise_variance = image_norm.var()

    # Calculate the expected variance of the noise at each timestep
    # Note: In a diffusion model, the variance at timestep t can be estimated based on the beta schedule
    expected_variances = (1 - alpha_cumprod) * noise_variance  # Adjust as necessary

    # Find the closest expected variance to the actual noise variance
    estimated_timestep = (torch.abs(expected_variances - noise_variance)).argmin()

    print(estimated_timestep)
    # Ensure the timestep is within a valid range
    return max(1, min(estimated_timestep.item(), len(betas) - 1))  # Convert to int value




import torch

def denoise(model, noisy_image, timestep, beta_schedule, device="cuda"):
    """
    Denoise an image using a pre-trained DDPM model, starting from a specific timestep.

    Parameters:
    - model: The trained denoising model.
    - noisy_image: The noisy image tensor to be denoised.
    - timestep: The specific timestep to start denoising from.
    - beta_schedule: The beta schedule used during training.
    - device: Device to perform inference on (default: cuda:0).

    Returns:
    - current_image: The denoised image.
    """
    model.eval()  # Ensure model is in evaluation mode
    current_image = noisy_image.to(device)

    if current_image.dim() == 3:
        current_image = current_image.unsqueeze(0)  # Add batch dimension if missing

    # Ensure the timestep is valid
    if timestep < 0 or timestep >= len(beta_schedule):
        raise ValueError(f"Invalid timestep: {timestep}. Must be in range [0, {len(beta_schedule) - 1}]")

    with torch.no_grad():
        # Predict the noise using the model
        timestep_tensor = torch.full((1,), timestep, dtype=torch.long, device=device)
        noise_pred = model(current_image, timestep_tensor)

        # Compute reverse process parameters for the current timestep
        beta_t = beta_schedule[timestep]
        alpha_t = torch.cumprod(1 - beta_schedule[:timestep + 1], dim=0)[-1].to(device)
        alpha_t_bar_sqrt = alpha_t.sqrt()
        one_minus_alpha_t_bar_sqrt = (1 - alpha_t).sqrt()

        # Compute the denoised image step
        current_image = (current_image - one_minus_alpha_t_bar_sqrt * noise_pred) / alpha_t_bar_sqrt

        # Clamp the image values between 0 and 1
        current_image = torch.clamp(current_image, 0.0, 1.0)

    return current_image.cpu()


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model from snapshot
    model = load_model(config, config["training"]["snapshot_path"], device=device)

    # Load a test image
    test_image = utils.load_image_as_tensor("/home/jaltieri/ddpmx/dataset128/R_b344347d-2db8-432c-8ca7-f660af506286_160_unfiltered_frame_23.png", device=device)

    # Get beta schedule (same as used in training)
    beta_schedule = torch.tensor(utils.get_beta_schedule('linear', beta_start=0.0001, beta_end=0.006, num_diffusion_timesteps=100)).to(device)

    # Estimate the appropriate timestep based on image variance
    timestep = estimate_gaussian_timestep(test_image, beta_schedule)
    timestep=2
    # Denoise the image starting from the estimated timestep
    denoised_image = denoise(model, test_image, timestep, beta_schedule, device=device)

    # Save the denoised image to disk
    utils.save_image(denoised_image, "denoised_output2.png")

if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
