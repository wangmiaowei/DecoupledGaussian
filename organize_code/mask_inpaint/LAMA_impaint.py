import numpy as np
import cv2
from simple_lama_inpainting import SimpleLama
from PIL import Image
import argparse
import os

def main(sequence):
    
    img_folder = os.path.join("../input_dataset", sequence, "images")
    msk_folder = os.path.join("../input_dataset", sequence, "sam2_mask")
    painted_folder = os.path.join("../input_dataset", sequence, "painted_mask")
    # If not exist, create the mask folder
    os.makedirs(painted_folder, exist_ok=True)
    img_files = os.listdir(img_folder)
    for img_file in img_files:
            print(f"Processing {img_file} ...")
            # 1. Read the original image.
            image = Image.open(os.path.join(img_folder, img_file)).convert("RGB")
      
            # 2. Read the mask and convert it to a uint8 of 0/255.
            mask_np = np.load(os.path.join(msk_folder, 
                                           str(os.path.splitext(os.path.basename(img_file))[0])+'.npy'))
            assert mask_np.ndim == 2 and set(np.unique(mask_np)) <= {0, 1}
            mask_np = (mask_np * 255).astype(np.uint8)
      
            # 3. Define the expansion structure element (kernel) and the number of iterations
            kernel = np.ones((7, 7), np.uint8)
            mask_dilated = cv2.dilate(mask_np, kernel, iterations=6)
      
            # 4. Call inpainting and use the keyword argument to pass the mask
            result = simple_lama(image=image, mask=mask_dilated)
      
            # 5. Save the results
            result.save(os.path.join(painted_folder, img_file))

if __name__ == '__main__':
    simple_lama = SimpleLama()
    parser = argparse.ArgumentParser(description='Run LAMA to impaint on an image sequence')
    parser.add_argument('sequence', help='Sequence folder name under ../input_dataset')
    args = parser.parse_args()
    main(args.sequence)
