import os
from glob import glob
import cv2

ORIGINALS_DIR = '../data_unet/predict_input/'
MASKS_DIR = '../processed/cleaned_masks/'
OUTPUT_DIR = '../processed/rois/'

def extract(original_img, mask, save):
    img = cv2.imread(original_img)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"Cannot read {original_img} or {mask}")
        return
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype('uint8')
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(save, result)
    
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    supported_extensions = ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff']
    mask_paths = []
    for ext in supported_extensions:
        mask_paths.extend(glob(os.path.join(MASKS_DIR, f'*{ext}')))
    
    if not mask_paths:
        print(f"No mask found in {MASKS_DIR}")
        return

    print(f"{len(mask_paths)} mask found, extracting...")
    for mask_path in mask_paths:
        file_name = os.path.basename(mask_path)
        original_img_path = os.path.join(ORIGINALS_DIR, file_name)
        
        if not os.path.exists(original_img_path):
            print(f"WARNING: ORIGINAL IMAGE OF {file_name} NOT FOUND!")
            continue

        save_path = os.path.join(OUTPUT_DIR, file_name)
        extract(original_img_path, mask_path, save_path)

    print("Extracted!")

if __name__ == '__main__':
    main()


