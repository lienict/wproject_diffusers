import rembg
import torch
import numpy as np
from PIL import Image,ImageOps
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

# Load the input image
init_image = load_image("./3334.png").resize((512,768))
 
# Convert the input image to a numpy array
input_array = np.array(init_image)
 
# Extract mask using rembg
mask_array = rembg.remove(input_array, only_mask=True)
 
# Create a PIL Image from the output array
mask_image = Image.fromarray(mask_array)
mask_image_inverted = ImageOps.invert(mask_image)
 
# Display inverted mask
# mask_image_inverted
# Create inpainting pipeline
pipeline = AutoPipelineForInpainting.from_pretrained(
    "redstonehero/ReV_Animated_Inpainting", 
    torch_dtype=torch.float16
)
 
pipeline.enable_model_cpu_offload()
prompt = """forest
"""
 
negative_prompt = ""
 
image = pipeline(prompt=prompt,
             negative_prompt=negative_prompt,
             width=512,
             height=768,
             num_inference_steps=20,
             image=init_image, 
             mask_image=mask_image_inverted,
             guidance_scale=1,
             strength=0.7, 
             generator=torch.manual_seed(189018)).images[0]

# Save the result as a transparent PNG
with open("./output/test.jpg", "wb") as output_file:
    output_file.write(image)