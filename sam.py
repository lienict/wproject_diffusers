from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

def change_background_stable_diffusion(product_image_path, mask_path, prompt, output_path):
    # Load the Stable Diffusion inpainting model
    model_id = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")  # Use GPU if available
    pipe = pipe.to("cpu")

    # Load the product image and mask
    product_img = Image.open('./3334.png').convert("RGB")
    mask_img = Image.open('./3334.png').convert("L")  # Load mask in grayscale

    # Apply Stable Diffusion inpainting
    result = pipe(prompt=prompt, image=product_img, mask_image=mask_img).images[0]

    # Save the output image
    result.save(output_path)
    print(f"Saved the inpainted image with new background at {output_path}")

# Example usage
change_background_stable_diffusion(
    product_image_path="product_image.jpg",
    mask_path="background_mask.png",
    prompt="a beautiful forest background",  # Describe the desired background
    output_path="output_image.jpg"
)
