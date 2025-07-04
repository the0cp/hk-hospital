import sys
import os
from glob import glob

MODEL_PATH = '../models/unet/MODEL.pth' 
INPUT_IMAGE_DIR = '../data_unet/predict_input/'
OUTPUT_MASK_DIR = '../processed/predicted_masks/'

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: '{MODEL_PATH}'")
        sys.exit(1)

    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    supported_extensions = ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff']
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob(os.path.join(INPUT_IMAGE_DIR, f'*{ext}')))

    if not image_paths:
        print(f"No suitable images found")
        sys.exit(1)

    print(f"Found {len(image_paths)} images...")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_MASK_DIR, img_name)
        
        command = (f"python predict.py --model \"{MODEL_PATH}\" "
                   f"--input \"{img_path}\" "
                   f"--output \"{output_path}\"")
        
        print(f"Processing {img_name}...")
        os.system(command)

if __name__ == '__main__':
    main()