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


import torch

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
    
    for t in reversed(range(1, starting_timestep+1)):
        with torch.no_grad():
            eta_theta = model(x_t,torch.tensor([t],device=x_t.device))
        
        beta_t = beta_schedule[t]
        beta_t = torch.tensor(beta_t, device=x_t.device) if not isinstance(beta_t, torch.Tensor) else beta_t

        x_t = (1 / torch.sqrt(1 - beta_t)) * (x_t - beta_t * eta_theta)
        
    return x_t.cpu()

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model from snapshot
    model = load_model(config, config["training"]["snapshot_path"], device=device)

    beta_schedule = utils.get_beta_schedule('linear', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=100)
    betas = torch.from_numpy(beta_schedule).float().to(device)

    # Load a test image
    test_image = utils.load_image_as_tensor("/home/jaltieri/ddpmx/dataset128/R_b344347d-2db8-432c-8ca7-f660af506286_160_unfiltered_frame_23.png", device=device)

    # Estimate the appropriate timestep based on image variance

    timestep = estimate_timestep(test_image,betas,"gaussian")
    print(f"gaussian timestep: {timestep}")
   
    # Denoise the image starting from the estimated timestep
    denoised_image = denoise_image(model,test_image,beta_schedule,timestep)

    # Save the denoised image to disk
    utils.save_image(denoised_image, "denoised_output.png")


if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
