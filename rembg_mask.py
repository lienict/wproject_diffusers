from rembg import remove
from PIL import Image,ImageOps
import numpy as np

# Load the original image
image_path = "./dog.png"
original_image = Image.open(image_path).convert("RGBA")  # Load image with alpha channel for transparency

# Step 3: Use rembg to create a binary mask
# `rembg` returns an image with transparent background. We can create a binary mask from this.
output_image = remove(original_image)

# Convert the transparent background image to a mask
# Any non-transparent pixel is considered part of the foreground (white), background is black
mask = Image.new("L", output_image.size, 0)  # Create a new black (background) image
mask_data = np.array(output_image)[:, :, 3]  # Use the alpha channel to create the mask
mask = Image.fromarray((mask_data > 0).astype(np.uint8) * 255)  # Convert to binary mask (255 foreground, 0 background)

mask_image_inverted = ImageOps.invert(mask)

# Save the mask if needed
mask.save("./mask_image1.png")
mask_image_inverted.save("./mask_image_inverted.png")