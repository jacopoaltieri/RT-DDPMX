import yaml
import numpy as np
import os
import torch
from PIL import Image
from torch.distributed import init_process_group

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
    # Load the image as a 16-bit grayscale image
    image = Image.open(image_path).convert("I;16")
    # Normalize the image to the [0, 1] range
    image = np.array(image, dtype=np.float32) / 65535.0

    # Convert to a tensor and add a channel dimension to make it compatible with model input
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
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

    # Clip values to ensure they are within the range [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Convert the tensor to a NumPy array and scale it to 16-bit range [0, 65535]
    image_array = (image_tensor.numpy() * 65535).astype('uint16')

    # Create a PIL Image from the NumPy array
    image = Image.fromarray(image_array, mode='I;16')

    # Save the image as a 16-bit grayscale PNG
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
        
if __name__ == "__main__":
    pass