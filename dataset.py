import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PNGDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("L")  # Convert image to grayscale ("L" mode)
        
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform is provided, convert to tensor directly
            image = transforms.ToTensor()(image)
            
        return image



if __name__ == "__main__":
    pass