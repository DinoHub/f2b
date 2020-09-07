# import time
import numpy as np

class F2B:
    def __init__ (self, detect_fn=None, max_inference_width=None, max_inference_height=None, overlapx=None, overlapy=None, overlapx_px=None, overlapy_px=None, pad=False):
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

    def register(self, big_frame_shape):
        '''
        big_frame_shape: np-array like, [ h, w ]
        '''
        self.big_frame_h, self.big_frame_w = big_frame_shape
        print('Registered: Image {}x{} '.format(self.big_frame_w, self.big_frame_h) )
        if self.noslice:
            return 1
        else:
            print('Overlap(w,h): ', self.overlapx_px, self.overlapy_px)
            step_w = self.max_inference_width - self.overlapx_px
            step_h = self.max_inference_height - self.overlapy_px
            self.steps = [step_w, step_h]
            num_ws = int( 1 + np.ceil((self.big_frame_w-self.max_inference_width)/step_w) )
            num_hs = int( 1 + np.ceil((self.big_frame_h-self.max_inference_height)/step_h) )
            self.num_smols = [num_ws, num_hs]
            return num_ws * num_hs

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
            # numpy pad
            # tic = time.time()
            new_smol = np.zeros((self.max_inference_height, self.max_inference_width, 3), dtype='uint8')
            for i, mean in enumerate(self.rgb_means):
                new_smol[:,:,i] = np.pad(smol[:,:,i], [[0, bpad],[0,rpad]], mode='constant', constant_values=mean)
            # toc = time.time()
            # print('np pad padding time:{}s'.format(toc - tic))

            # concat
            # tic = time.time()
            # bot_pad = np.ones((bpad,w,3))
            # r_pad = np.ones((h+bpad,rpad,3))
            # for i, mean in enumerate(self.rgb_means):
            #     bot_pad[:,:,i]*=mean
            #     r_pad[:,:,i]*=mean
            # new_smol = np.concatenate((smol, bot_pad), axis=0)
            # new_smol = np.concatenate((new_smol, r_pad), axis=1)
            # toc = time.time()
            # print('concat padding time:{}s'.format(toc - tic))
            return new_smol
        else:
            return smol
    
    # def pad_big(self, big_frame):
    #     h, w = big_frame.shape[:2]
    #     new_big_frame_w = int ((self.num_smols[0] - 1) * self.steps[0] + self.max_inference_width)
    #     new_big_frame_h = int((self.num_smols[1] - 1) * self.steps[1] + self.max_inference_height)
    #     rpad =  new_big_frame_w - w
    #     bpad =  new_big_frame_h - h

    #     # numpy pad
    #     tic = time.time()
    #     new_big_frame = np.zeros((new_big_frame_h, new_big_frame_w, 3))
    #     for i, mean in enumerate(self.rgb_means):
    #         new_big_frame[:,:,i] = np.pad(big_frame[:,:,i], [[0, bpad],[0,rpad]], mode='constant', constant_values=mean)
    #     toc = time.time()
    #     print('numpy pad padding time:{}s'.format(toc - tic))

        # concat
        # tic = time.time()
        # bot_pad = np.ones((bpad,w,3))
        # r_pad = np.ones((h+bpad,rpad,3))
        # for i, mean in enumerate(self.rgb_means):
        #     bot_pad[:,:,i]*=mean
        #     r_pad[:,:,i]*=mean
        #     new_big_frame = np.concatenate((big_frame, bot_pad), axis=0)
        #     new_big_frame = np.concatenate((new_big_frame, r_pad), axis=1)
        # toc = time.time()
        # print('concat padding time:{}s'.format(toc - tic))
        
        # return new_big_frame2

    def get_start_w(self, i):
        return max( 0, 
                    int(i*self.steps[0])
                   )
    def get_end_w(self, i):
        return min( self.big_frame_w,
                    int(self.get_start_w(i)+self.max_inference_width)
                   )
    def get_start_h(self, i):
        return max( 0, 
                    int(i*self.steps[1])
                   )
    def get_end_h(self, i):
        return min( self.big_frame_h, 
                    int(self.get_start_h(i)+self.max_inference_height)
                   )

    def slice_n_dice(self, big_frame):
        '''
        Returns
            smol_frames : list of np-array like
            smol_coords : list of tuple, each tuple (l,t,r,b) top left point in global coordinate
        '''
        smol_frames = []
        smol_coords = []
        # big_frame = self.pad_big(big_frame)
        for j in range(self.num_smols[1]):
            for i in range( self.num_smols[0] ):
                h1 = self.get_start_h(j)
                h2 = self.get_end_h(j)
                w1 = self.get_start_w(i)
                w2 = self.get_end_w(i)
                smol = big_frame[h1:h2, w1:w2]
                smol = self.pad_smol(smol)
                smol_frames.append(smol)
                smol_coords.append((w1,h1, w2-1,h2-1))
        print('Num of smols: {}'.format(len(smol_frames)))
        return smol_frames, smol_coords

    def deconflicting_union(self, od_results, smol_coords):
        flatten_dets = []
        smol_indices = []
        # Converting all detections to global coordinate
        for i, res in enumerate(od_results):
            # il,it, ir, ib = smol_coords[i]
            il,it = smol_coords[i][:2]
            # iw = ir - il + 1
            # ih = ib - it + 1
            # scale_x = iw / self.max_inference_width
            # scale_x = 1
            # scale_y = ih / self.max_inference_height
            # scale_y = 1
            for det in res:
                # l,t,w,h = det[0]
                l,t,r,b = det[0]
                # l = l * scale_x + il
                # t = t * scale_y + it
                # w = w * scale_x
                # h = h * scale_y
                l = l + il
                t = t + it
                r = r + il
                b = b + it
                # new_det = [ l, t, w, h ]
                new_det = [ l, t, r, b ]
                flatten_dets.append((new_det, det[1], det[2]))
                smol_indices.append(i)
        return flatten_dets, smol_indices


    def detect(self, big_frame,  **kwargs):
        if self.noslice:
            res = self.detect_fn(big_frame, **kwargs)
            return res, [0 for _ in res], [( 0,0, self.big_frame_w-1, self.big_frame_w-1 )]
        else:
            smol_frames, smol_coords = self.slice_n_dice(big_frame)
            # box_format NEEDS to be in ltrb for f2b to process correctly, deconflicting_union assumes that. if your detect_fn does not have this argument, please make the necessary changes.
            od_results = self.detect_fn(smol_frames, box_format='ltrb', **kwargs)

            # select = -2
            # import cv2
            # last_smol = smol_frames[select].copy()
            # last_res = od_results[select]
            # for res in last_res:
            #     l,t,w,h = res[0]
            #     r = l + w - 1
            #     b = t + h - 1
            #     print(res)
            #     cv2.rectangle(last_smol, (l,t), (r,b), (255,0,0), 2)
            # cv2.imshow('last', last_smol)
            # cv2.waitKey(0)

            flatten_dets, smol_indices = self.deconflicting_union(od_results, smol_coords)
            return flatten_dets, smol_indices, smol_coords

    # def detect2(self, big_frame, **kwargs):
    #     if self.noslice:
    #         return self.detect_fn(big_frame, **kwargs)
    #     else:
    #         smol_frames, smol_coords = self.slice_n_dice(big_frame)
    #         for i, frame in enumerate(smol_frames):
    #             od_results = self.detect_fn(frame, **kwargs)
    #             il,it, ir, ib = smol_coords[i]
    #             iw = ir - il + 1
    #             ih = ib - it + 1
    #             scale_x = iw / self.max_inference_width
    #             scale_y = ih / self.max_inference_height
    #             frame_show = frame.copy()
    #             for det in od_results:
    #                 l,t,w,h = det[0]
    #                 l = int(l * scale_x)
    #                 t = int(t * scale_y)
    #                 w = w * scale_x
    #                 h = h * scale_y
    #                 r = int(l + w - 1)
    #                 b = int(t + h - 1)
    #                 cv2.rectangle(frame_show, (l,t), (r,b), (255,255,0), 2)
    #             cv2.imwrite('f2b/{}_det.jpg'.format(i), frame_show)        
    #             print(od_results)
    #         exit()                    

# if __name__ == '__main__':
#     # import cv2
#     # f2b = F2B(max_inference_width=5, max_inference_height=5)
#     # f2b = F2B(max_inference_width=1920, max_inference_height=1080, pad=True)
#     # f2b = F2B(max_inference_width=1920, max_inference_height=1080)
#     # f2b = F2B(max_inference_width=1920, inference_AR=16/9)
#     # f2b = F2B(max_inference_width=1280, max_inference_height=720)
#     # biggie = np.random.rand(13,13,3) #height, width, channel
#     # biggie = cv2.imread('/media/dh/HDD/4K_sea_scenes/DJI_0044_4K_SEA_decoded/DJI_0044_4K_SEA_frame0186.jpg')
  
#     # import time
#     # tic = time.time()  
#     # smols = f2b.register(biggie.shape[:2]) 
#     # toc = time.time()
#     # print('reg time:{}s'.format(toc - tic))
#     # tic = time.time()  
#     # smols = f2b.slice_n_dice(biggie) 
#     # toc = time.time()
#     # print('slice and dice time:{}s'.format(toc - tic))

#     # # print(biggie)
#     # # import pdb; pdb.set_trace()
#     # for i, smol in enumerate(smols):
#     #     cv2.imwrite('{}.jpg'.format(i), smol)
#     #     print('Smol size: {}x{}'.format(smol.shape[1], smol.shape[0]))
#     #     # cv2.imshow('{}'.format(i), smol)
#     # cv2.waitKey()

#     import cv2
#     biggie = cv2.imread('/media/dh/HDD/4K_sea_scenes/DJI_0044_4K_SEA_decoded/DJI_0044_4K_SEA_frame0186.jpg')
#     import sys
#     sys.path.append('..')
#     from kerasyolo.yolo import YOLO
#     od = YOLO(bgr=False, gpu_usage=0.5, score=0.5,
#               batch_size=8,
#               input_image_size=(1080, 1920),
#               model_path='kerasyolo/model_data/pp_reanchored_best_val.h5',
#               anchors_path='kerasyolo/model_data/PP_ALL_anchors.txt',
#               classes_path='kerasyolo/model_data/PP_classes.txt')

#     # print(biggie.shape)
#     # print(type(biggie))
#     # print(biggie.dtype)
#     # res=od.detect_get_box_in([biggie, biggie], box_format='ltwh', classes=['ship'], buffer_ratio=0.0)
#     # print(res)

#     # f2b = F2B(max_inference_width=-1, inference_AR=16/9, detect_fn=od.detect_get_box_in, pad=True)
#     # f2b = F2B(max_inference_width=1920, inference_AR=16/9, detect_fn=od.detect_get_box_in, pad=True)
#     f2b = F2B(max_inference_width=od.input_image_size[1], max_inference_height=od.input_image_size[0], detect_fn=od.detect_get_box_in, pad=False)
#     # f2b = F2B(max_inference_width=1920, inference_AR=16/9, detect_fn=od.detect_get_box_in, pad=False)
#     num_smols = f2b.register(biggie.shape[:2]) 
#     # od.regenerate(num_smols)
#     # f2b.detect2(biggie, box_format='ltwh', classes=['ship'], buffer_ratio=0.0)
#     flatten_dets, smol_indices, smol_coords = f2b.detect(biggie, box_format='ltwh', classes=['ship'], buffer_ratio=0.0)
#     import random
#     def draw(frameDC, bbs, smol_coords, smol_indices):
#         if bbs is None or len(bbs) == 0:
#             return

#         font = cv2.FONT_HERSHEY_DUPLEX
#         fontScale = 1.2
#         fontThickness = 2
#         frame_h, frame_w = frameDC.shape[:2]
#         colors = [ (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(len(smol_coords)) ]

#         for i, coords in enumerate(smol_coords):
#             l,t,w,h = coords
#             r = l + w - 1
#             b = t + h - 1
#             cv2.rectangle(frameDC, (l,t), (r,b), colors[i], 4)
#         for i, bb in enumerate(bbs):
#             if bb is None:
#                 continue
#             l,t,w,h = [ int(x) for x in bb[0]]
#             r = l + w - 1
#             b = t + h - 1
#             text = bb[2]
#             color = colors[smol_indices[i]]
#             cv2.rectangle(frameDC, (l,t), (r,b), color, 2)
#             cv2.putText(frameDC, 
#                         text, 
#                         (l+5, b-10),
#                         font, fontScale, color, fontThickness)
#             # if t - 10 - self.indTextSize[1] >= 0: 
#             #     text_y = int(t - 10)
#             # elif b + 10 + self.indTextSize[1] <= frame_h - 1:
#             #     text_y = int(b + 10 + self.indTextSize[1])
#             # else:
#             #     text_y = int(t + (b-t)/2 + self.indTextSize[1]/2)

#             # cv2.putText(frameDC, 
#                         # str(i), 
#                         # (l+5, text_y),
#                         # font, fontScale*2.5, color, fontThickness*2)
#     biggie_show = biggie.copy()
#     draw(biggie_show, flatten_dets, smol_coords, smol_indices)
#     show_win_name = 'biggie'
#     cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)
#     cv2.imshow(show_win_name, biggie_show)
#     # cv2.imwrite('Biggie.jpg', biggie_show)
#     cv2.waitKey(0)