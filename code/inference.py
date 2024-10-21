import torch
from models import UNet
import utils

def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load a pre-trained model from a .pt file.

    Parameters:
    - model_path: The path to the model file (.pt).
    - device: The device to load the model onto (e.g., "cuda" or "cpu").

    Returns:
    - model: The loaded model.
    """
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

    state_dict = torch.load(model_path, map_location=device)

    if 'MODEL_STATE' in state_dict:
        model.load_state_dict(state_dict['MODEL_STATE'])
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

def estimate_gaussian_timestep(image, max_timesteps=1000):
    noise_variance = image.var().item()
    threshold = 1e-6
    if noise_variance < threshold:
        return 1
    alpha_t = min(1.0, noise_variance / (noise_variance + threshold))
    timestep = int(max_timesteps * (1 - alpha_t))
    timestep = max(1, min(timestep, max_timesteps))
    return timestep

def denoise(model, noisy_image, timestep, device="cuda:0"):
    model.eval()
    current_image = noisy_image.to(device)

    if current_image.dim() == 3:
        current_image = current_image.unsqueeze(0)

    timestep_tensor = torch.full((1,), timestep, dtype=torch.int, device=device)

    with torch.no_grad():
        for t in range(timestep, -1, -1):
            timestep_tensor.fill_(t)
            noise_pred = model(current_image, timestep_tensor)
            mu_theta = current_image - noise_pred
            current_image = mu_theta + torch.randn_like(current_image) * torch.sqrt(torch.tensor(1 - (t / timestep)))
            current_image = torch.clamp(current_image, 0.0, 1.0)

    return current_image.cpu()

def main(config):
    device = 'cuda'
    model = load_model(config["training"]["snapshot_path"], device=device)

    test_image = utils.load_image_as_tensor("/home/jaltieri/ddpmx/dataset128/R_b344347d-2db8-432c-8ca7-f660af506286_160_unfiltered_frame_23.png", device=device)
    timestep = estimate_gaussian_timestep(test_image)
    denoised_image = denoise(model, test_image, timestep, device=device)

    utils.save_image(denoised_image, "denoised_output.png")

if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
