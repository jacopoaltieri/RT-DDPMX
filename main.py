import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PNGDataset
import utils
from tqdm import tqdm
import models
import losses


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    # Load config
    CFG_PATH = r"./cfg.yaml"
    cfg = utils.load_yaml(CFG_PATH)

    dataset = PNGDataset(cfg["dataset"], transform=None)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    unet = models.UNet().to(DEVICE)
    opt = optim.AdamW(unet.parameters(), lr = cfg["lr"])
    scheduler =  optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["cos_annealing_t_max"])
    scaler = torch.GradScaler(str(DEVICE))
    
    def train_epoch(unet, dataloader, optimizer, scaler, epoch):
        unet.train()
        running_loss = 0.0
        
        for batch_idx, images in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg["epochs"]}")):
            images = images.to(DEVICE)
            batch_size = images.size(0)
            timesteps = torch.randint(0, 1000, (batch_size,), device=DEVICE)  # Random timesteps

            # Forward pass
            with torch.autocast(str(DEVICE)):
                noisy_images = images + torch.randn_like(images) * 0.1  # Simulate noisy data
                outputs = unet(noisy_images, timesteps)
                loss = losses.mse_loss(outputs, images)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{cfg["epochs"]}, Loss: {avg_loss:.4f}")

    def evaluate(unet, dataloader):
        unet.eval()
        with torch.no_grad():
            for batch_idx, images in enumerate(tqdm(dataloader, desc="Evaluation")):
                images = images.to(DEVICE)
                timesteps = torch.randint(0, 1000, (images.size(0),), device=DEVICE)  # Random timesteps
                outputs = unet(images, timesteps)
                
                # Save some outputs for inspection
                if batch_idx == 0:
                    save_image(outputs.cpu(), 'outputs.png', nrow=4)
                    

    for epoch in range(cfg["epochs"]):
        train_epoch(unet, dataloader, opt, scaler, epoch)
        if (epoch + 1) % 10 == 0:
            evaluate(unet, dataloader)
    print("Training complete!")