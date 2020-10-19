import random
import colorsys

import cv2
import numpy as np

font = cv2.FONT_HERSHEY_DUPLEX

fontScale = 1.2
fontThickness = 2
boxThickness = 2

# fontScale = 0.75
# fontScale = 1.0
# fontThickness = 1
# boxThickness = 1

text_buff = 0 #px


def draw_biggie(frameDC, bbs, smol_coords, smol_indices, conf_thresh=None):
    if bbs is None or len(bbs) == 0:
        return

    # font = cv2.FONT_HERSHEY_DUPLEX
    # fontScale = 1.2
    # fontThickness = 2
    # frame_h, frame_w = frameDC.shape[:2]

    colors = [ tuple( [c*255 for c in colorsys.hsv_to_rgb(h,1,1)] ) 
                    for h in np.linspace(0, 1, num=len(smol_coords)+1) ]
                    # for h in np.linspace(0, 1, num=len(smol_coords)) ]
    # colors = [ tuple([c*255 for c in color ]) for color in colors ]
    # colors = [ (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(len(smol_coords)) ]

    for i, coords in enumerate(smol_coords):
        # l,t,w,h = coords
        l,t,r,b = coords
        # r = l + w - 1
        # b = t + h - 1
        cv2.rectangle(frameDC, (l,t), (r,b), colors[i], 4)
        # print(l,t,r,b)
    
    for i, bb in enumerate(bbs):
        if bb is None:
            continue
        conf = bb[1]
        if conf_thresh is not None and conf < conf_thresh:
            continue

        l,t,r,b = [ int(x) for x in bb[0]]
        # r = l + w - 1
        # b = t + h - 1

        text = f'{bb[2]}:{conf*100:0.0f}%'
        color = colors[smol_indices[i]]
        cv2.rectangle(frameDC, (l,t), (r,b), color, boxThickness)

        text_size, _ = cv2.getTextSize(text, font, fontScale, fontThickness)
        text_w, text_h = text_size 

        cv2.putText(frameDC, 
                    text, 
                    (l+5, b + text_h + text_buff),
                    font, fontScale, color, fontThickness)


def draw_dets(frameDC, bbs, conf_thresh=None):
    if bbs is None or len(bbs) == 0:
        return

    # font = cv2.FONT_HERSHEY_DUPLEX
    # fontScale = 1.2
    # fontThickness = 2

    color = (255,255,0)
    
    for i, bb in enumerate(bbs):
        if bb is None:
            continue
        conf = bb[1]
        if conf_thresh is not None and conf < conf_thresh:
            continue

        l,t,r,b = [ int(x) for x in bb[0]]
        text = f'{bb[2]}:{conf*100:0.0f}%'
        cv2.rectangle(frameDC, (l,t), (r,b), color, boxThickness)
        
        text_size, _ = cv2.getTextSize(text, font, fontScale, fontThickness)
        text_w, text_h = text_size 

        cv2.putText(frameDC, 
                    text, 
                    (l+5, b+text_h+text_buff),
                    font, fontScale, color, fontThickness)

def draw_smol(frameDC, smol_coords):
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.2
    fontThickness = 2

    colors = [ tuple( [c*255 for c in colorsys.hsv_to_rgb(h,1,1)] )
                    for h in np.linspace(0, 1, num=len(smol_coords)) ]

    for i, coords in enumerate(smol_coords):
        l,t,r,b = coords
        cv2.rectangle(frameDC, (l,t), (r,b), colors[i], 4)
        # print(l,t,r,b)
