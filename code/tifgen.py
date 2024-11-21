import os
from PIL import Image

# Set the directory containing the images
input_directory = "/home/jaltieri/ddpmx/output_512"
output_directory ="./tifs_512"
os.makedirs(output_directory, exist_ok=True)  # Create the output folder if it doesn't exist

# Dictionary to group images by their base name (e.g., "imagex")
image_groups = {}

# Iterate through files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust based on your image formats
        # Extract the base name and frame number
        base_name = filename.split('_frame')[0]
        
        # Open the image and group by base name
        img_path = os.path.join(input_directory, filename)
        if base_name not in image_groups:
            image_groups[base_name] = []
        image_groups[base_name].append((filename, Image.open(img_path)))

# Create a TIFF file for each group of images
for base_name, image_list in image_groups.items():
    if image_list:
        # Sort the images by frame number extracted from the filename
        sorted_images = sorted(
            image_list,
            key=lambda x: int(x[0].split('_frame_')[1].split('_')[0])  # Extract numeric frame number
        )
        
        # Extract the sorted images only (discard filenames after sorting)
        sorted_images = [img[1] for img in sorted_images]
        
        # Use the first frame's filename for naming the TIFF
        first_frame_filename = os.path.basename(image_list[0][0])
        tiff_filename = f"{first_frame_filename.split('_frame')[0]}.tiff"
        tiff_path = os.path.join(output_directory, tiff_filename)  # Save to the output directory
        
        # Save as a multi-frame TIFF
        sorted_images[0].save(
            tiff_path,
            save_all=True,
            append_images=sorted_images[1:],
            compression="tiff_lzw"
        )
        print(f"Saved {tiff_path} with {len(sorted_images)} frames.")

print("All images have been grouped, sorted, and saved as TIFF files.")