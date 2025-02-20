import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Manual Model Loading ---
# Specify the local path to your model file.
MODEL_PATH = r"E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-BRIA_AI-RMBG\RMBG-1.4"  # Change this path to where your model is stored

try:
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
except Exception as e:
    print("Error loading model from local drive:", e)
    raise e

class CutPNGNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI image tensor
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
        """
        Convert the ComfyUI tensor to a PIL RGB image.
        Expected input tensor shape: (1, C, H, W) with values in [0,1].
        """
        image_array = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(image_array).convert("RGB")
        return pil_img

    def get_mask(self, pil_img):
        """
        Resize the image to the model's input size, run the model to get a segmentation mask,
        and return the binary mask as a PIL grayscale image resized to the original image size.
        """
        model_input_size = (320, 320)
        transform = transforms.Compose([
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
        ])
        input_tensor = transform(pil_img).unsqueeze(0)  # shape: (1, 3, 320, 320)

        with torch.no_grad():
            # Run the model; expecting a single-channel mask output
            mask_pred = model(input_tensor)[0]

        # Ensure mask is 2D
        if mask_pred.dim() == 3 and mask_pred.shape[0] == 1:
            mask_pred = mask_pred.squeeze(0)
        mask_np = mask_pred.cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.uint8) * 255  # Convert to binary (0 or 255)
        mask_pil = Image.fromarray(binary_mask, mode="L")
        # Resize the mask back to original image size
        mask_pil = mask_pil.resize(pil_img.size, Image.BILINEAR)
        return mask_pil

    def cut_image(self, image, x, y, width, height):
        """
        1. Convert the input ComfyUI image tensor to a PIL image.
        2. Get a segmentation mask using the manually loaded model.
        3. Convert the original image to RGBA and apply the mask as the alpha channel.
        4. Crop the image to the specified coordinates.
        5. Convert the result back to a tensor.
        """
        # Convert input tensor to a PIL image
        pil_img = self.preprocess_image(image)
        
        # Get the segmentation mask (binary mask)
        mask = self.get_mask(pil_img)
        
        # Convert the original image to RGBA (to allow transparency)
        pil_rgba = pil_img.convert("RGBA")
        result = Image.new("RGBA", pil_rgba.size)
        result.paste(pil_rgba, mask=mask)  # Use the mask as the alpha channel
        
        # Crop the result image
        cropped_result = result.crop((x, y, x + width, y + height))
        
        # Convert the cropped image back to a tensor
        result_np = np.array(cropped_result).astype(np.float32) / 255.0  # shape: (H, W, 4)
        result_tensor = torch.tensor(result_np).permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
        return (result_tensor,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "CutPNGNode": CutPNGNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutPNGNode": "RMBG Cut PNG"
}
