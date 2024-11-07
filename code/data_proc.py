import os
import cv2
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor
import utils


def process_image(file_path, output_folder, roi_size):
    # Read the multi-page TIFF image
    tiff = cv2.imreadmulti(file_path, [], cv2.IMREAD_UNCHANGED)

    if not tiff[0]:
        print(f"Failed to read {file_path}")
        return

    frames = tiff[1]
    num_frames = len(frames)

    # Process only the middle frame
    middle_frame_index = num_frames // 2
    frame = frames[middle_frame_index]

    # Calculate the number of ROIs along the width and height
    frame_height, frame_width = frame.shape
    roi_height, roi_width = roi_size
    num_rois_x = frame_width // roi_width
    num_rois_y = frame_height // roi_height

    # Loop to extract each ROI and save it
    for y in range(num_rois_y):
        for x in range(num_rois_x):
            # Extract ROI
            x_start = x * roi_width
            y_start = y * roi_height
            roi = frame[y_start:y_start + roi_height, x_start:x_start + roi_width]

            # Construct the output file name with frame and ROI indices
            output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{middle_frame_index + 1}_roi_{y + 1}_{x + 1}.png"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Save the ROI as a 16-bit PNG
            cv2.imwrite(output_file_path, roi)
            print(f"Saved {output_file_path}")


def process_folder(input_folder, output_folder, roi_size, max_workers):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tiff') or f.endswith('.tif')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path in files:
            executor.submit(process_image, file_path, output_folder, roi_size)


if __name__ == "__main__":
    
    CFG_PATH = r"./cfg.yaml"
    
    num_cores = os.cpu_count()
    max_workers = min(num_cores // 2, 4)
    
    cfg = utils.load_yaml(CFG_PATH)
    
    input_folder = cfg["dataset"]["seq_folder"]
    output_folder = cfg["dataset"]["directory"]
    img_size = tuple(cfg["dataset"]["input_size"])

    process_folder(input_folder, output_folder, img_size, max_workers)
