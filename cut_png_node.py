import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Download and load the BRIA RMBG v1.4 model.
# This will download the model file (a TorchScript model) from Hugging Face.
MODEL_FILE = hf_hub_download(repo_id="briaai/RMBG-1.4", filename="rmbg-1.4.pt")
model = torch.jit.load(MODEL_FILE, map_location="cpu")
model.eval()

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
        Converts the ComfyUI tensor to a PIL RGB image.
        """
        # Convert from (1, C, H, W) with values in [0,1] to (H, W, C) uint8
        image_array = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(image_array).convert("RGB")
        return pil_img

    def get_mask(self, pil_img):
        """
        Resizes the image to the model's input size, runs the model to get a segmentation mask,
        and then returns the mask as a PIL grayscale image resized to the original image size.
        """
        model_input_size = (320, 320)
        transform = transforms.Compose([
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
        ])
        input_tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 320, 320)

        with torch.no_grad():
            # The model outputs a segmentation mask (assumed to be a single channel)
            mask_pred = model(input_tensor)[0]

        if mask_pred.dim() == 3 and mask_pred.shape[0] == 1:
            mask_pred = mask_pred.squeeze(0)

        mask_np = mask_pred.cpu().numpy()
        # Threshold to create a binary mask
        binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
        mask_pil = Image.fromarray(binary_mask, mode="L")
        # Resize the mask to the original image size
        mask_pil = mask_pil.resize(pil_img.size, Image.BILINEAR)
        return mask_pil

    def cut_image(self, image, x, y, width, height):
        """
        Processes the input image:
        1. Converts the ComfyUI tensor to a PIL image.
        2. Obtains a segmentation mask using the RMBG model.
        3. Applies the mask to remove the background (creates transparency).
        4. Crops the resulting image to the specified region.
        5. Converts the result back to a ComfyUI tensor.
        """
        # Step 1: Preprocess the input image
        pil_img = self.preprocess_image(image)
        
        # Step 2: Obtain the segmentation mask using the RMBG model
        mask = self.get_mask(pil_img)
        
        # Step 3: Convert the original image to RGBA to support transparency
        pil_rgba = pil_img.convert("RGBA")
        
        # Create a new image for output (with a transparent background)
        result = Image.new("RGBA", pil_rgba.size)
        result.paste(pil_rgba, mask=mask)  # Apply the mask as the alpha channel
        
        # Step 4: Crop the image based on provided coordinates
        cropped_result = result.crop((x, y, x + width, y + height))
        
        # Step 5: Convert the cropped image back to a tensor for ComfyUI
        result_np = np.array(cropped_result).astype(np.float32) / 255.0  # shape: (H, W, 4)
        result_tensor = torch.tensor(result_np).permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
        return (result_tensor,)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "CutPNGNode": CutPNGNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutPNGNode": "RMBG Cut PNG"
}
