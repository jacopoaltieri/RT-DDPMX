import os
import re
import time
import torch
import numpy as np
from torch.amp import autocast
from tqdm import tqdm
from collections import defaultdict
import utils


def detect_input_type(input_folder: str):
    """
    Detect whether the input folder contains ROIs or full images.
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")

    roi_pattern = re.compile(r"(.+_unfiltered_frame)")
    base_names = set()

    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            if "rescaled" in image_file.lower():
                return 'full'
            if roi_pattern.match(image_file):
                base_names.add(image_file)

    return 'rois' if base_names else 'full'


def estimate_timestep(image: torch.Tensor, betas: torch.Tensor, noise_type: str ="gaussian") -> int:
    """
    Estimate the timestep based on the noise variance of a noisy image.
    """
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # Normalize image to [0, 1] intensity range for variance estimation
    image_norm = (image - image.min()) / (image.max() - image.min())
    noise_variance = image_norm.var().item()

    if noise_type.lower() == "gaussian":
        expected_variances = 1 - alpha_cumprod
    
    # FIXME    
    elif noise_type.lower() == "poisson":
        expected_variances = (1 - alpha_cumprod) * noise_variance
    else:
        raise ValueError("noise_type must be 'gaussian' or 'poisson'")

    estimated_timestep = torch.abs(expected_variances - noise_variance).argmin()
    return max(1, min(estimated_timestep.item(), len(betas) - 1))


def estimate_average_timestep_for_image(rois, betas, noise_type="gaussian"):
    """
    Estimate average timestep based on the noise variance of all ROIs.
    """
    timesteps = [estimate_timestep(roi, betas, noise_type) for roi in rois]
    return int(round(np.mean(timesteps)))


def denoise_image(model, noisy_images, beta_schedule, starting_timestep):
    x_t = noisy_images
    for t in reversed(range(1, starting_timestep + 1)):
        with torch.no_grad():
            with autocast(device_type=x_t.device.type):
                eta_theta = model(x_t, torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.int64))

        beta_t = beta_schedule[t]
        beta_t = torch.tensor(beta_t, device=x_t.device) if not isinstance(beta_t, torch.Tensor) else beta_t
        x_t.sub_(beta_t * eta_theta).div_(torch.sqrt(1 - beta_t)) # x_t = (1 / torch.sqrt(1 - beta_t)) * (x_t - beta_t * eta_theta) , but faster
    return x_t.cpu()


def process_single_image_folder(model, betas, input_folder: str, output_folder: str, device):
    os.makedirs(output_folder, exist_ok=True)
    beta_tensor = torch.tensor(betas, dtype=torch.float32, device=device)

    for image_file in tqdm(sorted(os.listdir(input_folder)), desc="Processing images"):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(input_folder, image_file)
        image = utils.load_image_as_tensor(image_path, device=device,add_batch=True)
        avg_timestep = estimate_average_timestep_for_image([image], beta_tensor, "gaussian")
        print(f"Estimated timestep for {image_file}: {avg_timestep}")

        start_time = time.time()
        denoised_image = denoise_image(model, image, beta_tensor, avg_timestep)

        output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_denoised.png")
        utils.save_image(denoised_image.squeeze(0), output_file_path)
        print(f"Denoised image saved to: {output_file_path}")

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        torch.cuda.empty_cache()


def process_folder_with_avg_t(model, betas, input_folder: str, output_folder: str, device):
    os.makedirs(output_folder, exist_ok=True)
    image_groups = defaultdict(list)
    pattern = re.compile(r"(.+_unfiltered_frame)")

    for image_file in sorted(os.listdir(input_folder)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            match = pattern.match(image_file)
            if match:
                image_groups[match.group(1)].append(os.path.join(input_folder, image_file))

    beta_tensor = torch.tensor(betas, dtype=torch.float32, device=device)
    for base_name, image_paths in tqdm(image_groups.items(), desc="Processing ROI groups"):
        rois = [utils.load_image_as_tensor(path, device=device) for path in image_paths]
        rois_batch = torch.stack(rois).to(device)

        avg_timestep = estimate_average_timestep_for_image(rois, beta_tensor, "gaussian")
        print(f"Average timestep for {base_name}: {avg_timestep}")

        start_time = time.time()
        denoised_images = denoise_image(model, rois_batch, beta_tensor, avg_timestep)

        for idx, image_path in enumerate(image_paths):
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_denoised.png")
            utils.save_image(denoised_images[idx], output_file_path)
            print(f"Saved: {output_file_path}")

        print(f"Group '{base_name}' processing time: {time.time() - start_time:.2f} seconds")
        torch.cuda.empty_cache()

def main(config):
    input_folder = "/home/jaltieri/ddpmx/rois256"
    output_folder = "/home/jaltieri/ddpmx/output_256_64"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = utils.load_model(config, config["training"]["snapshot_path"], device=device)
    beta_schedule = utils.get_beta_schedule('linear', beta_start=0.00001, beta_end=0.02, num_diffusion_timesteps=100)

    input_type = detect_input_type(input_folder)
    print(f"Detected input type: {input_type}")

    if input_type == 'rois':
        process_folder_with_avg_t(model, beta_schedule, input_folder, output_folder, device)
    elif input_type == 'full':
        process_single_image_folder(model, beta_schedule, input_folder, output_folder, device)
    else:
        print("Unable to determine input type.")

if __name__ == "__main__":
    config = utils.load_yaml("cfg_256.yaml")
    main(config)
