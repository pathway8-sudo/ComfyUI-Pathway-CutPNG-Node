import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class CutPNGNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  
                "x": ("INT", {"default": 0, "min": 0}),
                "y": ("INT", {"default": 0, "min": 0}),
                "width": ("INT", {"default": 256, "min": 1}),
                "height": ("INT", {"default": 256, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cut_image"
    CATEGORY = "Image Processing"

    def preprocess_image(self, image):
        """Convert a ComfyUI tensor to a PIL RGB image."""
        image_array = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        return Image.fromarray(image_array).convert("RGB")

    def cut_image(self, image, x, y, width, height):
        """Crop an image based on given coordinates and return the processed tensor."""
        pil_img = self.preprocess_image(image)
        cropped_result = pil_img.crop((x, y, x + width, y + height))
        result_np = np.array(cropped_result).astype(np.float32) / 255.0
        result_tensor = torch.tensor(result_np).permute(2, 0, 1).unsqueeze(0)  
        return (result_tensor,)

NODE_CLASS_MAPPINGS = {
    "CutPNGNode": CutPNGNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutPNGNode": "Cut PNG (Background Remove)"
}
