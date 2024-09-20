import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class PNGDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load the image as a 16-bit grayscale image
        image = Image.open(img_path).convert("I;16")
        # Normalize the image to [0, 1] range
        image = np.array(image) / 65535.0  # Normalize 16-bit range to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Generate a random timestep (or define a method to load specific timesteps)
        timestep = torch.randint(0, 1000, (1,)).item()  # Assuming timesteps are in the range [0, 999]
        
        return image, timestep
    
if __name__ == "__main__":
    pass