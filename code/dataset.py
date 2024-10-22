import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import utils


config = utils.load_yaml("cfg.yaml")
max_timesteps = config["model"]["max_timesteps"]


class PNGDataset(Dataset):
    def __init__(self, image_dir, max_timesteps=max_timesteps):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.max_timesteps = max_timesteps  # Maximum number of timesteps

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load the image as a 16-bit grayscale image
        image = Image.open(img_path).convert("I;16")
        
        # Normalize the image to the [0, 1] range
        image = np.array(image, dtype=np.float32) / 65535.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Randomly sample a timestep between 1 and max_timesteps for training
        timestep = torch.randint(0, self.max_timesteps, (1,),  dtype=torch.long).item()
        
        return image, timestep

    
if __name__ == "__main__":
    pass