import random
import colorsys

import cv2
import numpy as np

def draw_biggie(frameDC, bbs, smol_coords, smol_indices):
    if bbs is None or len(bbs) == 0:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.2
    fontThickness = 2
    # frame_h, frame_w = frameDC.shape[:2]

    colors = [ tuple( [c*255 for c in colorsys.hsv_to_rgb(h,1,1)] ) 
                    for h in np.linspace(0, 1, num=len(smol_coords)) ]
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
        l,t,r,b = [ int(x) for x in bb[0]]
        # r = l + w - 1
        # b = t + h - 1
        text = f'{bb[2]}:{bb[1]*100:0.0f}%'
        color = colors[smol_indices[i]]
        cv2.rectangle(frameDC, (l,t), (r,b), color, 2)
        cv2.putText(frameDC, 
                    text, 
                    (l+5, b-10),
                    font, fontScale, color, fontThickness)


def draw_dets(frameDC, bbs):
    if bbs is None or len(bbs) == 0:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.2
    fontThickness = 2

    color = (255,255,0)
    
    for i, bb in enumerate(bbs):
        if bb is None:
            continue
        l,t,r,b = [ int(x) for x in bb[0]]
        text = f'{bb[2]}:{bb[1]*100:0.0f}%'
        cv2.rectangle(frameDC, (l,t), (r,b), color, 2)
        cv2.putText(frameDC, 
                    text, 
                    (l+5, b-10),
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
