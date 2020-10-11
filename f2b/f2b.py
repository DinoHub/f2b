from copy import deepcopy
from collections import defaultdict

# import time
import numpy as np

from .utils.box_utils import point_in_box, box_in_box, iou, box_intersections, box_union 

def get_sorted_tup(a,b):
    return tuple(sorted([a,b]))

def merge(detA, detB):
    '''
    det in [ ltrb, confidence, class ] format
    '''
    ltrbA, confA, clA = detA
    ltrbB, confB, clB = detB
    assert clA == clB
    return [ box_union(ltrbA, ltrbB) , max(confA, confB), clA ]

class F2B:
    def __init__ (self, detect_fn=None, max_inference_width=None, max_inference_height=None, overlapx=None, overlapy=None, overlapx_px=None, overlapy_px=None, pad=False, dcu=True, merge_thresh=0.85):
        self.detect_fn = detect_fn
        self.max_inference_width = max_inference_width if max_inference_width is not None else 1920
        # self.inference_AR = inference_AR 
        self.max_inference_height = max_inference_height if max_inference_height is not None else 1080
        # if self.max_inference_height is None and self.inference_AR: 
            # self.max_inference_height = int(self.max_inference_width / self.inference_AR)
        # else:
            # self.max_inference_height = 1080
        if self.max_inference_width <= 0 or self.max_inference_height <= 0:
            self.noslice = True
            print('Not slicing! Frame not big?')
        else:
            self.noslice = False
            print('Inference size (WxH): {}x{}'.format(self.max_inference_width, self.max_inference_height))

        if overlapx_px is None:
            self.overlapx = overlapx if overlapx is not None else 0.2 # Overlap ratio for width
            self.overlapx_px = self.max_inference_width * self.overlapx # in px
        else:
            self.overlapx_px = overlapx_px

        if overlapy_px is None:
            self.overlapy = overlapy if overlapy is not None else 0.2 # Overlap ratio for height
            self.overlapy_px = self.max_inference_width * self.overlapy # in px
        else:
            self.overlapy_px = overlapy_px
        
        self.rgb_means = [123.68, 116.779, 103.939] #imagenet
        self.pad = pad
        self.do_dcu = dcu
        self.merge_thresh = merge_thresh

    def get_start_x(self, i):
        return max( 0, 
                    int(i*self.steps[0])
                   )
    def get_end_x(self, i):
        return min( self.big_frame_w,
                    int(self.get_start_x(i)+self.max_inference_width)
                   )
    def get_start_y(self, i):
        return max( 0, 
                    int(i*self.steps[1])
                   )
    def get_end_y(self, i):
        return min( self.big_frame_h, 
                    int(self.get_start_y(i)+self.max_inference_height)
                   )

    def get_overlap_left_idx(self, slice_idx):
        row_idx = slice_idx % self.num_smol_ws 
        if row_idx == 0:
            return None
        else:
            return slice_idx - 1

    def get_overlap_right_idx(self, slice_idx):
        row_idx = slice_idx % self.num_smol_ws 
        if row_idx == (self.num_smol_ws - 1):
            return None
        else:
            return slice_idx + 1

    def get_overlap_top_idx(self, slice_idx):
        col_idx = slice_idx // self.num_smol_ws 
        if col_idx == 0:
            return None
        else:
            return slice_idx - self.num_smol_ws

    def get_overlap_bot_idx(self, slice_idx):
        col_idx = slice_idx // self.num_smol_ws 
        if col_idx == (self.num_smol_hs - 1):
            return None
        else:
            return slice_idx + self.num_smol_ws

    def register(self, big_frame_shape):
        '''
        big_frame_shape: np-array like, [ h, w ]
        '''
        self.big_frame_h, self.big_frame_w = big_frame_shape
        print('Registered: Image {}x{} '.format(self.big_frame_w, self.big_frame_h) )
        if self.noslice:
            self.smol_coords = [( 0,0, self.big_frame_w-1, self.big_frame_w-1 )]
            return 1
        else:
            print('Overlap(w,h): ', self.overlapx_px, self.overlapy_px)
            step_w = self.max_inference_width - self.overlapx_px
            step_h = self.max_inference_height - self.overlapy_px
            self.steps = [step_w, step_h]
            num_ws = int( 1 + np.ceil((self.big_frame_w-self.max_inference_width)/step_w) )
            num_hs = int( 1 + np.ceil((self.big_frame_h-self.max_inference_height)/step_h) )
            # self.num_smols = [num_ws, num_hs]
            self.num_smol_ws = num_ws
            self.num_smol_hs = num_hs

            self.smol_coords = []

            # self.overlap_x_regions = []
            # self.overlap_y_regions = []
            # nest_dict = lambda: defaultdict(dict)
            self.slice2overlaps = {}

            last_x2 = None
            last_y2 = None
            slice_index = 0
            for j in range(num_hs):
                for i in range( num_ws ):
                    x1 = self.get_start_x(i)
                    x2 = self.get_end_x(i)
                    y1 = self.get_start_y(j)
                    y2 = self.get_end_y(j)
                    self.smol_coords.append((x1,y1, x2-1,y2-1))

                    self.slice2overlaps[slice_index] = {}
                    if 0 < i < num_ws:
                        self.slice2overlaps[slice_index]['l'] = [(x1,y1, last_x2-1, last_y2-1), self.get_overlap_left_idx(slice_index)]
                        # self.overlap_x_regions.append((x1,y1, last_x2-1, last_y2-1))
                    if 0 < j < num_hs:
                        self.slice2overlaps[slice_index]['t'] = [(x1,y1, last_x2-1, last_y2-1), self.get_overlap_top_idx(slice_index)]
                        # self.overlap_y_regions.append((x1,y1, last_x2-1, last_y2-1))
                    last_x2 = x2
                    last_y2 = y2
                    slice_index += 1

            for slice_idx, overlap_items in self.slice2overlaps.items():
                right_idx = self.get_overlap_right_idx(slice_idx)
                if right_idx is not None:
                    overlap_items['r'] = [ self.slice2overlaps[right_idx]['l'][0], right_idx ]
                bot_idx = self.get_overlap_bot_idx(slice_idx)
                if bot_idx is not None:
                    overlap_items['b'] = [ self.slice2overlaps[bot_idx]['t'][0], bot_idx ]

            num_smols = num_ws * num_hs
            print('Num of smols: {}'.format(num_smols))
            return num_smols

    def need_pad(self, h, w):
        if self.pad and (h < self.max_inference_height or w < self.max_inference_width):
            return True
        else:
            # assert h == self.max_inference_height and w == self.max_inference_width
            return False

    def pad_smol(self, smol):
        h, w = smol.shape[:2]
        if self.need_pad(h,w):
            bpad = self.max_inference_height - h
            rpad = self.max_inference_width - w
            new_smol = np.zeros((self.max_inference_height, self.max_inference_width, 3), dtype='uint8')
            for i, mean in enumerate(self.rgb_means):
                new_smol[:,:,i] = np.pad(smol[:,:,i], [[0, bpad],[0,rpad]], mode='constant', constant_values=mean)
            return new_smol
        else:
            return smol
    
    def slice_n_dice(self, big_frame):
        '''
        Returns
            smol_frames : list of np-array like
            smol_coords : list of tuple, each tuple (l,t,r,b) top left point in global coordinate
        '''
        smol_frames = []

        for coords in self.smol_coords:
            x1, y1, x2, y2 = coords
            # print(y1, y2+1)
            # print(x1, x2+1)
            smol = big_frame[y1:y2+1, x1:x2+1]
            smol = self.pad_smol(smol)
            # print(smol.shape)
            smol_frames.append(smol)

        # return smol_frames, smol_coords
        return smol_frames


    # def deconflicting_union(self, od_results, smol_coords):
    def deconflicting_union(self, od_results):
        
        from pprint import pprint
    
        flatten_dets = []
        smol_indices = []

        od_results_global = []
        dets_only_global = []
        # Converting to global coordinate
        for slice_idx, res in enumerate(od_results):
            il,it = self.smol_coords[slice_idx][:2]
            slice_res_global = []
            slice_dets_only_global = []
            for det in res:
                l,t,r,b = det[0]
                l = l + il
                t = t + it
                r = r + il
                b = b + it
                new_det = [ l, t, r, b ]
                new_res = (new_det, det[1], det[2])
                flatten_dets.append(new_res)
                smol_indices.append(slice_idx)
                slice_res_global.append(new_res)
                slice_dets_only_global.append(new_det)
            od_results_global.append(slice_res_global)
            dets_only_global.append(slice_dets_only_global)

        if not self.do_dcu:
            return flatten_dets, smol_indices
        # Else continue our journey
        
        dets_tuples_in_overlaps_x = defaultdict(dict)
        dets_tuples_in_overlaps_y = defaultdict(dict)
        slice2intersect_idxes = defaultdict(dict)
        # Finding boxes in overlap regions
        for slice_idx, slice_dets in enumerate(dets_only_global):
            if len(slice_dets) == 0:
                continue

            overlap_regions = []
            overlap_directions = ''

            for direction, region in self.slice2overlaps[slice_idx].items():
                overlap_regions.append(region[0])
                overlap_directions += direction

            if overlap_directions == '':
                continue

            overlap_regions_np = np.array(overlap_regions)
            slice_dets_np = np.array(slice_dets)
            intersect_bools, intersections = box_intersections(overlap_regions_np, slice_dets_np)

            for direction, this_intersections, this_bools in zip(overlap_directions, intersections, intersect_bools):
                intersect_idxes = np.where(this_bools)[0]
                # intersections_candidates = this_intersections[intersect_idxes]
                
                # slice2intersect_idxes[slice_idx][direction] = intersect_idxes
                
                overlap_slice_idx = self.slice2overlaps[slice_idx][direction][1]


                # organise into classes
                class2dets = defaultdict(list)
                class2idxes = defaultdict(list)
                for idx in intersect_idxes:
                    cl = od_results_global[slice_idx][idx][2]
                    class2dets[cl].append(this_intersections[idx])
                    class2idxes[cl].append(idx)

                for cl, intersct_cands in class2dets.items():
                    intersct_cands = np.array(intersct_cands)
                    if direction == 'r':
                        dets_tuples_in_overlaps_x[get_sorted_tup(slice_idx, overlap_slice_idx)][cl] = [intersct_cands, None]
                    elif direction == 'l':
                        if cl in dets_tuples_in_overlaps_x[get_sorted_tup(slice_idx, overlap_slice_idx)]:
                            dets_tuples_in_overlaps_x[get_sorted_tup(slice_idx, overlap_slice_idx)][cl][1]= intersct_cands
                    elif direction == 'b':
                        dets_tuples_in_overlaps_y[get_sorted_tup(slice_idx, overlap_slice_idx)][cl] = [intersct_cands, None]
                    elif direction == 't':
                        if cl in dets_tuples_in_overlaps_y[get_sorted_tup(slice_idx, overlap_slice_idx)]:
                            dets_tuples_in_overlaps_y[get_sorted_tup(slice_idx, overlap_slice_idx)][cl][1]= intersct_cands
                    else:
                        raise Exception('Direction not supported')
        
                slice2intersect_idxes[slice_idx][direction] = class2idxes

        # pprint(slice2intersect_idxes)

        # Merge dem overlaps

        merged_dict = {}
        ## horizontally first
        # pprint(dets_tuples_in_overlaps_x)
        for slice_idxes, class_dict in dets_tuples_in_overlaps_x.items():
            sliceA, sliceB = slice_idxes
            for cl, pair_of_dets in class_dict.items():
                if pair_of_dets[1] is not None:
                    iou_matrix = iou(*pair_of_dets)
                    # print(slice_idxes)
                    # print('IOU', iou_matrix)
                    merger_idxes = np.argwhere(iou_matrix > self.merge_thresh)

                    for idxA, idxB in merger_idxes:
                        global_idxA = slice2intersect_idxes[sliceA]['r'][cl][idxA]
                        global_idxB = slice2intersect_idxes[sliceB]['l'][cl][idxB]
                        detA = od_results_global[sliceA][global_idxA]
                        detB = od_results_global[sliceB][global_idxB]
                        od_results_global[sliceA][global_idxA] = merge(detA, detB)

                        merged_dict[(sliceB,global_idxB)] = (sliceA, global_idxA)

        ## then vertically
        # pprint(dets_tuples_in_overlaps_y)
        for slice_idxes, class_dict in dets_tuples_in_overlaps_y.items():
            sliceA, sliceB = slice_idxes
            for cl, pair_of_dets in class_dict.items():
                if pair_of_dets[1] is not None:
                    iou_matrix = iou(*pair_of_dets)
                    # print(slice_idxes)
                    # print('IOU', iou_matrix)
                    merger_idxes = np.argwhere(iou_matrix > self.merge_thresh)

                    for idxA, idxB in merger_idxes:
                        global_idxA = slice2intersect_idxes[sliceA]['b'][cl][idxA]
                        global_idxB = slice2intersect_idxes[sliceB]['t'][cl][idxB]

                        if (sliceA, global_idxA) in merged_dict:
                            sliceA_resolved, global_idxA_resolved = merged_dict[(sliceA, global_idxA)]
                        else:
                            sliceA_resolved, global_idxA_resolved = sliceA, global_idxA
                        detA = od_results_global[sliceA_resolved][global_idxA_resolved]

                        if (sliceB, global_idxB) in merged_dict:
                            sliceB_resolved, global_idxB_resolved = merged_dict[(sliceB, global_idxB)]
                        else:
                            sliceB_resolved, global_idxB_resolved = sliceB, global_idxB
                        detB = od_results_global[sliceB_resolved][global_idxB_resolved]

                        od_results_global[sliceA_resolved][global_idxA_resolved] = merge(detA, detB)

                        # using sliceB instead of sliceB_resolved is correct. sliceB_resolved tuple already have a "pointer" to sliceA_resolved. Now we want to establish that this "new" sliceB should point to sliceA_resolved as well. 
                        merged_dict[(sliceB,global_idxB)] = (sliceA_resolved, global_idxA_resolved)


        flatten_dets = [ det for slice_idx, slice_dets in enumerate(od_results_global)
                             for in_slice_idx, det in enumerate(slice_dets) 
                             if (slice_idx, in_slice_idx) not in merged_dict   ]

        return flatten_dets, smol_indices


    def detect(self, big_frame,  **kwargs):
        '''
        Assumes detect_fn takes in list of frames, and output detections in [ ltrb, confidence, class  ] format.
        '''
        if self.noslice:
            res = self.detect_fn(big_frame, **kwargs)
            return res, [0 for _ in res]
        else:
            smol_frames = self.slice_n_dice(big_frame)
            # smol_frames, smol_coords = self.slice_n_dice(big_frame)
            # box_format NEEDS to be in ltrb for f2b to process correctly, deconflicting_union assumes that. if your detect_fn does not have this argument, please make the necessary changes.
            if len(smol_frames) > 0:
                od_results = self.detect_fn(smol_frames, box_format='ltrb', **kwargs)
                flatten_dets, smol_indices = self.deconflicting_union(od_results)
                return flatten_dets, smol_indices
            else:        
                return None, None

    def merge(self, big_frame, od_results):
        smol_frames = self.slice_n_dice(big_frame)
        # smol_frames, smol_coords = self.slice_n_dice(big_frame)
        # box_format NEEDS to be in ltrb for f2b to process correctly, deconflicting_union assumes that. if your detect_fn does not have this argument, please make the necessary changes.
        if len(smol_frames) > 0:
            flatten_dets, smol_indices = self.deconflicting_union(od_results)
            return flatten_dets, smol_indices
        else:
            return None, None
