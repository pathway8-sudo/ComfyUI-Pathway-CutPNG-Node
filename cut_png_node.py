import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

class CutPNGNode:
    def __init__(self):
        self.model = None  
        self.current_model_path = None  

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": r"E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\CutPNGNode\RMBG-1.4.pt"}),
                "x": ("INT", {"default": 0, "min": 0}),
                "y": ("INT", {"default": 0, "min": 0}),
                "width": ("INT", {"default": 256, "min": 1}),
                "height": ("INT", {"default": 256, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cut_image"
    CATEGORY = "Image Processing"

    def load_model(self, model_path):
        """Load RMBG model dynamically if it's not already loaded."""
        if self.model is None or self.current_model_path != model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ Model not found: {model_path}")

            try:
                self.model = torch.jit.load(model_path, map_location="cpu")
                self.model.eval()
                self.current_model_path = model_path
                print(f"✅ Model Loaded: {model_path}")
            except Exception as e:
                raise RuntimeError(f"❌ Error loading model: {e}")

    def preprocess_image(self, image):
        """Convert a ComfyUI tensor to a PIL RGB image."""
        image_array = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        return Image.fromarray(image_array).convert("RGB")

    def get_mask(self, pil_img):
        """Generate a binary mask using the RMBG model."""
        if self.model is None:
            raise RuntimeError("❌ RMBG Model is not loaded!")

        model_input_size = (320, 320)  
        transform = transforms.Compose([
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
        ])
        input_tensor = transform(pil_img).unsqueeze(0)  

        # Run the model
        with torch.no_grad():
            mask_pred = self.model(input_tensor)

        # Fix shape issues
        if mask_pred.dim() == 4:
            mask_pred = mask_pred.squeeze(0)
        if mask_pred.dim() == 3 and mask_pred.shape[0] == 1:
            mask_pred = mask_pred.squeeze(0)

        # Convert to binary mask
        mask_np = mask_pred.cpu().numpy().astype(np.float32)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]
        binary_mask = (mask_np > 0.5).astype(np.uint8) * 255

        return Image.fromarray(binary_mask, mode="L").resize(pil_img.size, Image.BILINEAR)

    def cut_image(self, image, model_path, x, y, width, height):
        """Load model, remove background, and crop the image."""
        self.load_model(model_path)  

        pil_img = self.preprocess_image(image)
        mask = self.get_mask(pil_img)

    
        pil_rgba = pil_img.convert("RGBA")
        result = Image.new("RGBA", pil_rgba.size)
        result.paste(pil_rgba, mask=mask)

        
        cropped_result = result.crop((x, y, x + width, y + height))
        result_np = np.array(cropped_result).astype(np.float32) / 255.0
        result_tensor = torch.tensor(result_np).permute(2, 0, 1).unsqueeze(0)

        return (result_tensor,)

NODE_CLASS_MAPPINGS = {
    "CutPNGNode": CutPNGNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutPNGNode": "Cut PNG (Remove Background)"
}
