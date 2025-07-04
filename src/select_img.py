import os
from pathlib import Path
masks_dir = Path('F:/RDphotos-2021-12-14_selectdata/1205manualmask')
images_dir = Path('F:/RDphotos-2021-12-14_selectdata/select_predictmask/test')


img_file = list(images_dir.glob('*'))

#mask_file = list(masks_dir.glob( '.*'))
#print(str(img_file[0])[45:52])
for i in img_file:
    x = str(i)[58:65]
 #   print(x)
#    print('copy "F:\RDphotos-2021-12-14_selectdata\1205manualmask\' + x +'_ROI.jpg"'+' "F:\RDphotos-2021-12-14_selectdata\select_mask"')
#    os.system('copy "F:\\RDphotos-2021-12-14_selectdata\\1205oriimg\\' + x +'_img.jpg"'+' "F:\\RDphotos-2021-12-14_selectdata\\918_origimg"')
 #   print('copy "F:\\RDphotos-2021-12-5\\interesting_area_manualmask\\test\\' + x +'_img.jpg"'+' "F:\\RDphotos-2021-12-14_selectdata\\select_generatemask"')
#    os.system('copy "F:\\RDphotos-2021-12-14_selectdata\\1205manualmask\\' + x + '_ROI.jpg"' + ' "F:\\RDphotos-2021-12-14_selectdata\\918manualmask"')