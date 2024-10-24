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

def estimate_gaussian_timestep(image:torch.Tensor, betas:torch.Tensor):
    """
    Estimate the timestep based on the variance of the noisy image and the beta schedule.
    
    Parameters:
    - image: Noisy image tensor.
    - betas: Beta values scheduled during training.

    Returns:
    - timestep: Estimated diffusion timestep.
    """
    
    
    alphas = 1- betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    image_norm = (image - image.min()) / (image.max() - image.min())

    # Calculate the variance of the noisy image
    noise_variance = image_norm.var()

    # Calculate the expected variance of the noise at each timestep
    # Note: In a diffusion model, the variance at timestep t can be estimated based on the beta schedule
    expected_variances = (1 - alpha_cumprod) * noise_variance  # Adjust as necessary

    # Find the closest expected variance to the actual noise variance
    estimated_timestep = (torch.abs(expected_variances - noise_variance)).argmin()

    print(estimated_timestep)
    # Ensure the timestep is within a valid range
    return max(1, min(estimated_timestep.item(), len(betas) - 1))  # Convert to int value



class GaussianDiffusion:
    
    def __init__(self, *, betas, device):
        self.device = device
        self.betas = betas.to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
    
    def diffuse(self, x0, t):
        assert t <= self.betas.shape[0]
        
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
            
        x0 = torch.tensor(x0).to(self.device)
        eps = torch.randn_like(x0).to(self.device)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        
        xt = torch.sqrt(alpha_cumprod_t)*x0 + \
            torch.sqrt(1-alpha_cumprod_t)*eps
        
        return xt
    
    def denoise(self, xt, eps, t):
        '''
        predict x0 from eps
        '''
        assert xt.shape == eps.shape
        
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
        
        xt = torch.tensor(xt).to(self.device)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        
        # estimate x0 from the predicted epsilon and a estimated t
        x0_pred = 1/torch.sqrt(alpha_cumprod_t)*xt - \
                torch.sqrt(1-alpha_cumprod_t)/torch.sqrt(alpha_cumprod_t)*eps 
        
        return x0_pred.detach().cpu()
    
    def dist_compare(self, xt, eps, t):
        assert xt.shape == eps.shape
        
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
        
        xt = torch.tensor(xt).to(self.device)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        
        # estimate x0 from the predicted epsilon and a estimated t
        target = 1/torch.sqrt(alpha_cumprod_t)*xt 
        predict = torch.sqrt(1-alpha_cumprod_t)/torch.sqrt(alpha_cumprod_t)*eps 
        
        return target, predict
    
    def reverse(self, xt, eps, t):
        '''
        predict x_{t-1} from x_{t} and eps
        '''
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
        
        beta_t = self.betas.index_select(0,t)
        alpha_t = self.alphas.index_select(0,t)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        alpha_cumprod_prev = self.alphas_cumprod.index_select(0,t-1)
        
        c1 = torch.sqrt(alpha_cumprod_prev)*beta_t / 1-alpha_cumprod_prev
        c2 = torch.sqrt(alpha_t)*(1-alpha_cumprod_prev) / 1-alpha_cumprod_t
        
        x0_pred = self.denoise(xt, eps, t)
        x_prev = c1*x0_pred + c2*xt
        
        return x_prev
    
    @staticmethod
    def _to_nparray_(x):
        x = x[0,0,:,:].detach().cpu().numpy()

        return x[:,:500]




def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model from snapshot
    model = load_model(config, config["training"]["snapshot_path"], device=device)

    beta_schedule = utils.get_beta_schedule('linear', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=100)
    betas = torch.from_numpy(beta_schedule).float().to(device)

    sample = GaussianDiffusion(betas=betas,device=device)
    # Load a test image
    test_image = utils.load_image_as_tensor("/home/jaltieri/ddpmx/dataset128/R_b344347d-2db8-432c-8ca7-f660af506286_160_unfiltered_frame_23.png", device=device)

    # Estimate the appropriate timestep based on image variance
    timestep = estimate_gaussian_timestep(test_image, betas)
    timestep= 9
    timestep = torch.tensor([timestep], device=device)  # Wrap in a list to make it 1D

    
    eps_t = model(test_image, timestep)  


    
    # Denoise the image starting from the estimated timestep
    denoised_image = sample.denoise(test_image,eps_t,timestep)
    # denoised_image = denoise(model, test_image, timestep, beta_schedule, device=device)

    # Save the denoised image to disk
    utils.save_image(denoised_image, "denoised_output2.png")

if __name__ == "__main__":
    config = utils.load_yaml("cfg.yaml")
    main(config)
