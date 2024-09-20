import yaml
import torch

# Read configuration from .YAML file
def load_yaml(path):
    with open(path) as file:
        try:
            cfg = yaml.safe_load(file)
            return cfg
        except yaml.YAMLError as exc:
            print(exc)
    
# Define a mock dataloader 
def generate_mock_dataloader(batch_size=2, image_size=(1, 256, 256), num_batches=10):
    for _ in range(num_batches):
        # Generate random images and timesteps
        images = torch.randn(batch_size, *image_size)
        timesteps = torch.randint(0, 1000, (batch_size,))
        yield images, timesteps    
    
if __name__ == "__main__":
    pass