import os
import cv2
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor
import utils



def process_image(file_path, output_folder, target_size):
    # Read the multi-page TIFF image
    tiff = cv2.imreadmulti(file_path, [], cv2.IMREAD_UNCHANGED)

    if not tiff[0]:
        print(f"Failed to read {file_path}")
        return

    frames = tiff[1]

    for i, frame in enumerate(frames):
        # Rescale the image to range [-1, 1]
        frame = frame.astype(np.float32)  # Convert to float32 for scaling
        frame = (frame / 32767.5) - 1  # Assuming the 16-bit images are unsigned (0-65535)

        # Resize the frame to the target size
        img_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        # Convert back to 16-bit integer (optional, for storage)
        img_resized = np.clip((img_resized + 1) * 32767.5, 0, 65535).astype(np.uint16)

        # Construct the output file name with frame index
        output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{i+1}.png"
        output_file_path = os.path.join(output_folder, output_file_name)

        # Save the resized frame
        cv2.imwrite(output_file_path, img_resized)

        print(f"Saved {output_file_path}")

def process_folder(input_folder, output_folder, target_size, max_workers):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tiff') or f.endswith('.tif')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path in files:
            executor.submit(process_image, file_path, output_folder, target_size)

if __name__ == "__main__":
    
    CFG_PATH = r"./cfg.yaml"
    
    num_cores = os.cpu_count()
    max_workers = min(num_cores // 2, 4)
    
    cfg = utils.load_yaml(CFG_PATH)
    
    input_folder = cfg["seq_folder"]
    output_folder = cfg["dataset"]
    img_size = tuple(cfg["input_size"])  # Convert list to tuple if needed

    process_folder(input_folder, output_folder, img_size, max_workers)
