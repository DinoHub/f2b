
def point_in_box(point, box):
    '''
    point: List[int], xy
    box: List[int], ltrb

    if point in box
    '''
    x,y = point
    l,t,r,b = box
    return l <= x <= r and t <= y <= b


def box_in_box(innerbox, outerbox):
    '''
    innerbox: List[int], ltrb
    outerbox: List[int], ltrb
    
    if innerbox is in outerbox
    '''

    point_lt = innerbox[:2]
    point_rb = innerbox[2:]

    return point_in_box(point_lt, outerbox) and point_in_box(point_rb, outerbox)


import numpy as np
def iou(boxes1, boxes2):
    '''
    From https://github.com/venuktan/Intersection-Over-Union
    boxes1: np.array( List[List[int]] ), [ [l1,t1,r1,b1], [l2,t2,r2,b2], ... ]
    boxes2: np.array( List[List[int]] ), [ [l1,t1,r1,b1], [l2,t2,r2,b2], ... ]
    '''
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def box_intersections(boxes1, boxes2):
    '''
    Adapted from above function
    boxes1: np.array( List[List[int]] ), [ [l1,t1,r1,b1], [l2,t2,r2,b2], ... ]
    boxes2: np.array( List[List[int]] ), [ [l1,t1,r1,b1], [l2,t2,r2,b2], ... ]
    '''

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    intersect_matrix = np.logical_and(xA <= xB, yA <= yB)
    intersections = np.dstack((xA, yA, xB, yB))

    return intersect_matrix, intersections

def box_union(boxA, boxB):
    l,t,r,b = boxA
    l2,t2,r2,b2 = boxB
    return [ min(l,l2), min(t,t2), max(r,r2), max(b,b2) ]


if __name__ == '__main__':
    import random 
    
    max_size = 500
    w = 200
    h = 100
    num_boxes1 = 5
    num_boxes2 = 5

    boxes1 = [ [ random.randint(0,max_size-1), random.randint(0,max_size-1) ] for _ in range(num_boxes1) ] 
    # boxes1 = [ [ box1[0], box1[1], box1[0]+w, box1[1]+h ] for box1 in boxes1  ]
    boxes1 = [ np.array([ box1[0], box1[1], box1[0]+w, box1[1]+h ]) for box1 in boxes1  ]

    boxes2 = [ [ random.randint(0,max_size-1), random.randint(0,max_size-1) ] for _ in range(num_boxes2) ] 
    boxes2 = [ [ box2[0], box2[1], box2[0]+w, box2[1]+h ] for box2 in boxes2  ]

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    print( iou(boxes1, boxes2) )

    intersect_matrix, intersections = box_intersections(boxes1, boxes2)
    print(intersect_matrix)
    # viz
    import cv2

    board = np.zeros((max_size+h, max_size+w, 3))

    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.2
    fontThickness = 2

    color = (255,255,0)
    for i, box in enumerate(boxes1):
        l,t,r,b = box
        text = f'{i}' 
        cv2.rectangle(board, (l,t), (r,b), color, thickness=1 )
        cv2.putText(board,text,(l+5, b-10), font, fontScale, color, fontThickness)

    color = (0,0,255)
    for i, box in enumerate(boxes2):
        l,t,r,b = box
        text = f'{i}' 

        cv2.rectangle(board, (l,t), (r,b), color, thickness=1 )
        cv2.putText(board,text,(l+5, b-10), font, fontScale, color, fontThickness)

    for box1_idx, zip_tup in enumerate(zip(intersect_matrix, intersections)):
        intersect_row, coord_row = zip_tup
        for box2_idx, zip_tup2 in enumerate(zip(intersect_row, coord_row)):
            intersect, coord = zip_tup2
            if intersect:
                l,t,r,b = coord
                text = f'{box1_idx}&{box2_idx}'
                cv2.rectangle(board, (l,t), (r,b), (255,255,255), thickness=-1 )
                cv2.putText(board,text,(r+10, b-10), font, fontScale*0.5, (255,255,255), 1)

    cv2.imshow('', board)
    cv2.waitKey(0)