import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class TIFSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.load_image_paths()

    def load_image_paths(self):
        """Load all TIF file paths in the root directory."""
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                self.image_paths.append(os.path.join(self.root_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load the TIF image sequence, extract frames, and apply transformations."""
        img_path = self.image_paths[idx]
        
        # Load the TIF image (multiframe) using cv2
        tif = cv2.imreadmulti(img_path, flags=cv2.IMREAD_ANYDEPTH)[1]

        # Normalize the 16-bit image to 0-1 range, then convert to float32
        frames = [frame.astype(np.float32) / 65535.0 for frame in tif]

        # Convert frames to a list of tensors
        frames = [torch.tensor(frame).unsqueeze(0) for frame in frames]

        # Apply transformations to each frame (if any)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return frames
    
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


if __name__ == "__main__":
    pass