import cv2

from f2b import F2B    

biggie = cv2.imread('/media/dh/HDD/4K_sea_scenes/DJI_0044_4K_SEA_decoded/DJI_0044_4K_SEA_frame0186.jpg')
print(biggie.shape)

# from kerasyolo.yolo import YOLO
# od = YOLO(bgr=False, gpu_usage=0.5, score=0.5,
#             batch_size=8,
#             input_image_size=(1080, 1920),
#             model_path='kerasyolo/model_data/pp_reanchored_best_val.h5',
#             anchors_path='kerasyolo/model_data/PP_ALL_anchors.txt',
#             classes_path='kerasyolo/model_data/PP_classes.txt')

from det2 import Det2
od = Det2(bgr=True, 
        weights= "det2/weights/faster-rcnn/faster_rcnn_R_50_FPN_3x/model_final_280758.pkl",
        config= "det2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        classes_path= 'det2/configs/coco80.names',
        thresh=0.5,
        )


f2b = F2B(inference_width= 1920, inference_height= 1080, detect_fn=od.detect_get_box_in, pad=False)



exit()

# f2b = F2B(inference_width=5, inference_height=5)
# f2b = F2B(inference_width=1920, inference_height=1080, pad=True)
# f2b = F2B(inference_width=1920, inference_height=1080)
# f2b = F2B(inference_width=1920, inference_AR=16/9)
# f2b = F2B(inference_width=1280, inference_height=720)
# biggie = np.random.rand(13,13,3) #height, width, channel
# biggie = cv2.imread('/media/dh/HDD/4K_sea_scenes/DJI_0044_4K_SEA_decoded/DJI_0044_4K_SEA_frame0186.jpg')

# import time
# tic = time.time()  
# smols = f2b.register(biggie.shape[:2]) 
# toc = time.time()
# print('reg time:{}s'.format(toc - tic))
# tic = time.time()  
# smols = f2b.slice_n_dice(biggie) 
# toc = time.time()
# print('slice and dice time:{}s'.format(toc - tic))

# # print(biggie)
# # import pdb; pdb.set_trace()
# for i, smol in enumerate(smols):
#     cv2.imwrite('{}.jpg'.format(i), smol)
#     print('Smol size: {}x{}'.format(smol.shape[1], smol.shape[0]))
#     # cv2.imshow('{}'.format(i), smol)
# cv2.waitKey()

# print(type(biggie))
# print(biggie.dtype)
# res=od.detect_get_box_in([biggie, biggie], box_format='ltwh', classes=['ship'], buffer_ratio=0.0)
# print(res)

# f2b = F2B(inference_width=-1, inference_AR=16/9, detect_fn=od.detect_get_box_in, pad=True)
# f2b = F2B(inference_width=1920, inference_AR=16/9, detect_fn=od.detect_get_box_in, pad=True)
f2b = F2B(inference_width=od.input_image_size[1], inference_height=od.input_image_size[0], detect_fn=od.detect_get_box_in, pad=False)
# f2b = F2B(inference_width=1920, inference_AR=16/9, detect_fn=od.detect_get_box_in, pad=False)
num_smols = f2b.register(biggie.shape[:2]) 
# od.regenerate(num_smols)
# f2b.detect2(biggie, box_format='ltwh', classes=['ship'], buffer_ratio=0.0)
flatten_dets, smol_indices, smol_coords = f2b.detect(biggie, box_format='ltwh', classes=['ship'], buffer_ratio=0.0)
import random
def draw(frameDC, bbs, smol_coords, smol_indices):
    if bbs is None or len(bbs) == 0:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.2
    fontThickness = 2
    frame_h, frame_w = frameDC.shape[:2]
    colors = [ (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(len(smol_coords)) ]

    for i, coords in enumerate(smol_coords):
        l,t,w,h = coords
        r = l + w - 1
        b = t + h - 1
        cv2.rectangle(frameDC, (l,t), (r,b), colors[i], 4)
    for i, bb in enumerate(bbs):
        if bb is None:
            continue
        l,t,w,h = [ int(x) for x in bb[0]]
        r = l + w - 1
        b = t + h - 1
        text = bb[2]
        color = colors[smol_indices[i]]
        cv2.rectangle(frameDC, (l,t), (r,b), color, 2)
        cv2.putText(frameDC, 
                    text, 
                    (l+5, b-10),
                    font, fontScale, color, fontThickness)
        # if t - 10 - self.indTextSize[1] >= 0: 
        #     text_y = int(t - 10)
        # elif b + 10 + self.indTextSize[1] <= frame_h - 1:
        #     text_y = int(b + 10 + self.indTextSize[1])
        # else:
        #     text_y = int(t + (b-t)/2 + self.indTextSize[1]/2)

        # cv2.putText(frameDC, 
                    # str(i), 
                    # (l+5, text_y),
                    # font, fontScale*2.5, color, fontThickness*2)
biggie_show = biggie.copy()
draw(biggie_show, flatten_dets, smol_coords, smol_indices)
show_win_name = 'biggie'
cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)
cv2.imshow(show_win_name, biggie_show)
# cv2.imwrite('Biggie.jpg', biggie_show)
cv2.waitKey(0)