try:
    from .cut_png_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    print("Error importing cut_png_node:", e)
    raise e
