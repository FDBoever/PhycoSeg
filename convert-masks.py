import os
import cv2
import numpy as np
from glob import glob

def convert_masks(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for mask_path in glob(os.path.join(input_dir, "*.png")):
        # Read with alpha channel (unchanged)
        mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask_rgba is None:
            print(f"could not read {mask_path}")
            continue

        if mask_rgba.shape[2] == 4:
            # Use alpha channel: non-transparent = foreground
            alpha = mask_rgba[:, :, 3]
            binary_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        else:
            # If no alpha, fallback: non-black pixels = foreground
            gray = cv2.cvtColor(mask_rgba, cv2.COLOR_BGR2GRAY)
            binary_mask = np.where(gray > 0, 255, 0).astype(np.uint8)

        # Save as 8-bit single-channel PNG
        filename = os.path.basename(mask_path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, binary_mask)

        print(f"saved {out_path}")

if __name__ == "__main__":
    # Example usage:
    convert_masks("raw_masks", "converted_masks")

