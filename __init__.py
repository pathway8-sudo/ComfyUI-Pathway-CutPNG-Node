from .cut_png_node import CutPNGNode  

NODE_CLASS_MAPPINGS = {
    "CutPNGNode": CutPNGNode  
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutPNGNode": "Cut PNG (Background Remove)"  
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
