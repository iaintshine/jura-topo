import os
from PIL import Image


class ImageDebugger:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.debug_dir = os.path.join(base_output_dir, "debug")
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def save_debug_image(self, image: Image.Image, name: str) -> None:
        debug_path = os.path.join(self.debug_dir, f"{name}.jpg")
        try:
            image.save(debug_path, "JPEG")
            print(f"Debug image saved to {debug_path}")
        except Exception as e:
            print(f"Failed to save debug image: {e}")
