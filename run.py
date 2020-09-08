import cv2
from pathlib import Path

from f2b import F2B    
from f2b import draw

input_image = 'examples/seoul-station_4K.png'
# input_image = 'examples/cars-pedestrian_4K.png'

biggie = cv2.imread(input_image)
# biggie = cv2.imread('examples/catdog_1080p.jpg')
print(biggie.shape)

from det2 import Det2
od = Det2(bgr=True, 
        weights= "det2/weights/faster-rcnn/faster_rcnn_R_50_FPN_3x/model_final_280758.pkl",
        config= "det2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        classes_path= 'det2/configs/coco80.names',
        thresh=0.5,
        max_batch_size=4,
        )

f2b = F2B(
        # max_inference_width = 0, 
        # max_inference_height = 0, 
        # max_inference_width = 1920, 
        # max_inference_height = 1080, 
        max_inference_width = 1333, 
        max_inference_height = 1333, 
        detect_fn=od.detect_get_box_in, 
        # overlapx = 0.2,
        # overlapy = 0.2,
        # overlapx_px = 200,
        # overlapy_px = 100,
        overlapx_px = 400,
        overlapy_px = 200,
        pad=False,
        dcu=True,
        merge_thresh=0.85
        )

num_smols = f2b.register(biggie.shape[:2]) 

flatten_dets, smol_indices = f2b.detect(biggie, classes=None, buffer_ratio=0.0)

biggie_show = biggie.copy()
draw.draw_biggie(biggie_show, flatten_dets, f2b.smol_coords, smol_indices)

show_win_name = 'biggie'
cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)
cv2.imshow(show_win_name, biggie_show)


input_path = Path(input_image)
out_path = Path('illustrations') / f'{input_path.stem}_det.jpg'
cv2.imwrite(str(out_path), biggie_show)

cv2.waitKey(0)
