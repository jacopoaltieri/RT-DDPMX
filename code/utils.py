import yaml
import numpy as np
import os
import torch
from PIL import Image
from torch.distributed import init_process_group
from models import UNet

def ddp_setup():
    """
    Set up the Distributed Data Parallel (DDP) environment.
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
    

def load_yaml(path):
    with open(path) as file:
        try:
            cfg = yaml.safe_load(file)
            return cfg
        except yaml.YAMLError as exc:
            print(exc)
    

def generate_mock_dataloader(batch_size=2, image_size=(1, 128, 128), num_batches=10):
    for _ in range(num_batches):
        # Generate random images and timesteps
        images = torch.randn(batch_size, *image_size)
        timesteps = torch.randint(0, 1000, (batch_size,))
        yield images, timesteps    
    
    
def load_image_as_tensor(image_path, device="cuda"):
    image = Image.open(image_path).convert("I;16")
    image = np.array(image, dtype=np.float32) / 65535.0

    # Convert to a tensor and add a channel dimension to make it compatible with model input
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # Shape: (1, 1, H, W)
    image_tensor = image_tensor.to(device)
    return image_tensor
    
    
def save_image(image_tensor, save_path):
    """
    Saves a tensor as a 16-bit grayscale PNG image.

    Args:
        image_tensor (torch.Tensor): The image tensor to save. Expected shape: (1, 1, H, W) or (1, H, W).
        save_path (str): The file path where the image will be saved.
    """
    # Remove batch dimension if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Shape: (1, H, W)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)  # Shape: (H, W)

    image_tensor = torch.clamp(image_tensor, 0, 1)
    image_array = (image_tensor.numpy() * 65535).astype('uint16')
    image = Image.fromarray(image_array, mode='I;16')
    image.save(save_path)
    print(f"Image saved to {save_path}")
    
    
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.array([
            beta_start + (beta_end - beta_start) * (t / (num_diffusion_timesteps - 1))
            for t in range(num_diffusion_timesteps)
        ], dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
 
 
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
    model.eval()
    return model 
 
        
if __name__ == "__main__":
    pass