import os
import cv2
import numpy as np
import re

def get_roi_position(filename):
    """Extract Y and X indices from the ROI filename."""
    match = re.search(r'_roi_(\d+)_(\d+)', filename)
    if match:
        y_index = int(match.group(1)) - 1  # Convert to 0-based index
        x_index = int(match.group(2)) - 1  # Convert to 0-based index
        return y_index, x_index
    return None

def patch_rois_together(base_name, roi_folder, output_folder):
    """Reconstruct the full image from ROIs based on the base name."""
    # Find all ROIs for the specified base image
    roi_files = [f for f in os.listdir(roi_folder) if f.startswith(base_name) and '_roi_' in f]
    if not roi_files:
        print(f"No ROIs found for base name {base_name} in {roi_folder}")
        return

    # Determine the grid size by finding the maximum Y and X indices
    max_y = max(get_roi_position(f)[0] for f in roi_files)
    max_x = max(get_roi_position(f)[1] for f in roi_files)

    # Read the first ROI to get the dimensions
    sample_roi = cv2.imread(os.path.join(roi_folder, roi_files[0]), cv2.IMREAD_UNCHANGED)
    roi_height, roi_width = sample_roi.shape

    # Create an empty array to hold the reconstructed image
    reconstructed_image = np.zeros(((max_y + 1) * roi_height, (max_x + 1) * roi_width), dtype=sample_roi.dtype)

    # Place each ROI in the correct position
    for roi_file in roi_files:
        y_index, x_index = get_roi_position(roi_file)
        roi_path = os.path.join(roi_folder, roi_file)
        roi = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)

        # Calculate the position in the reconstructed image
        y_start = y_index * roi_height
        y_end = y_start + roi_height
        x_start = x_index * roi_width
        x_end = x_start + roi_width

        # Place the ROI in the reconstructed image
        reconstructed_image[y_start:y_end, x_start:x_end] = roi

    # Save the reconstructed image
    output_path = os.path.join(output_folder, f"{base_name}_reconstructed.png")
    cv2.imwrite(output_path, reconstructed_image)
    print(f"Reconstructed image saved at {output_path}")

if __name__ == "__main__":
    # Parameters
    roi_folder = "/home/jaltieri/ddpmx/output_256sca"
    output_folder = "/home/jaltieri/ddpmx/den256sca"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all unique base names
    processed_basenames = set()
    for roi_file in os.listdir(roi_folder):
        base_name_match = re.match(r'^(.*?)_roi_\d+_\d+', roi_file)
        if base_name_match:
            base_name = base_name_match.group(1)
            if base_name not in processed_basenames:
                processed_basenames.add(base_name)
                patch_rois_together(base_name, roi_folder, output_folder)
