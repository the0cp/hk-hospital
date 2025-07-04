import os
import numpy as np
import cv2
from glob import glob
from skimage import morphology
#ori_mask = Image.open('F:/RDphotos-2021-12-5/generate_mask_resize/G2_0404_img.jpg')
#img_copy = np.copy(ori_mask)
#mask_ndarray = np.asarray(ori_mask)
#print(mask_ndarray.shape)
#mask_img = img_copy.astype(np.bool)

# post_mask = morphology.remove_small_holes(label(ori_mask),16384)

#post_mask = morphology.remove_small_objects(label(ori_mask))

#im = Image.fromarray(post_mask)
#im.save("post_mask.jpg")
#cv2.imwrite("post_mask.jpg", post_mask)
# 以灰


# 二值化，100为阈值，小于100的变为255，大于100的变为0
# 也可以根据自己的要求，改变参数：
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO_INV
# cv2.THRESH_TOZERO

# 找到所有的轮廓

INPUT_DIR = '../processed/predicted_masks/'
OUTPUT_DIR = '../processed/cleaned_masks/'

def post_process_mask(mask_path, save_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return
    processed_mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=2000, connectivity=2).astype(np.uint8)
    processed_mask *= 255
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros(processed_mask.shape, np.uint8)
        cv2.drawContours(final_mask, [max_contour], -1, 255, cv2.FILLED)
        cv2.imwrite(save_path, final_mask)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    supported_extensions = ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff']
    mask_paths = []
    for ext in supported_extensions:
        mask_paths.extend(glob(os.path.join(INPUT_DIR, f'*{ext}')))

    if not mask_paths:
        print(f"No mask found in {INPUT_DIR}")
        return

    for mask_path in mask_paths:
        file_name = os.path.basename(mask_path)
        save_path = os.path.join(OUTPUT_DIR, file_name)
        post_process_mask(mask_path, save_path)
    
if __name__ == '__main__':
    main()