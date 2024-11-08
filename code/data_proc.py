import os
import cv2
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor
import utils


def process_image(file_path, output_folder, roi_size, process_all_frames=False):
    # Read the multi-page TIFF image
    tiff = cv2.imreadmulti(file_path, [], cv2.IMREAD_UNCHANGED)

    if not tiff[0]:
        print(f"Failed to read {file_path}")
        return

    frames = tiff[1]
    num_frames = len(frames)

    # Determine frames to process
    frames_to_process = (
        [frames[num_frames // 2]] if not process_all_frames else frames
    )

    for frame_index, frame in enumerate(frames_to_process):
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
                frame_suffix = frame_index + 1 if process_all_frames else "middle"
                output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{frame_suffix}_roi_{y + 1}_{x + 1}.png"
                output_file_path = os.path.join(output_folder, output_file_name)

                # Save the ROI as a 16-bit PNG
                cv2.imwrite(output_file_path, roi)
                print(f"Saved {output_file_path}")


def rescale_image(file_path, output_folder, target_shape, process_all_frames=False):
    # Read the multi-page TIFF image
    tiff = cv2.imreadmulti(file_path, [], cv2.IMREAD_UNCHANGED)

    if not tiff[0]:
        print(f"Failed to read {file_path}")
        return

    frames = tiff[1]
    num_frames = len(frames)

    # Determine frames to process
    frames_to_process = (
        [frames[num_frames // 2]] if not process_all_frames else frames
    )

    for frame_index, frame in enumerate(frames_to_process):
        # Resize the frame to the target shape
        resized_frame = cv2.resize(frame, target_shape, interpolation=cv2.INTER_LINEAR)

        # Construct the output file name with frame index
        frame_suffix = frame_index + 1 if process_all_frames else "middle"
        output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{frame_suffix}_rescaled.png"
        output_file_path = os.path.join(output_folder, output_file_name)

        # Save the resized frame as a 16-bit PNG
        cv2.imwrite(output_file_path, resized_frame)
        print(f"Saved {output_file_path}")


def process_folder(input_folder, output_folder, img_size, max_workers, process_all_frames=False, mode="roi"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tiff') or f.endswith('.tif')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path in files:
            if mode == "roi":
                executor.submit(process_image, file_path, output_folder, img_size, process_all_frames)
            elif mode == "rescale":
                executor.submit(rescale_image, file_path, output_folder, img_size, process_all_frames)


if __name__ == "__main__":
    CFG_PATH = r"./cfg_512.yaml"
    
    num_cores = os.cpu_count()
    max_workers = min(num_cores // 2, 4)
    
    cfg = utils.load_yaml(CFG_PATH)
    
    input_folder = cfg["dataset"]["seq_folder"]
    output_folder = cfg["dataset"]["directory"]
    img_size = tuple(cfg["dataset"]["input_size"])
    process_all_frames = cfg["dataset"].get("process_all_frames", False)
    mode = cfg["dataset"].get("mode", "roi")  # Default to "roi" but can be "rescale"

    process_folder(input_folder, output_folder, img_size, max_workers, process_all_frames, mode)
