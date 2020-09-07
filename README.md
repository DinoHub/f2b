# F2B

## Frame Too Big

So we slice and dice, infer, then merge them back.

If you just stuff a 4K image into a detector:
![noslice](illustrations/seoul-station_4K_det_noslice.jpg)

After f2b:
![f2bed](illustrations/seoul-station_4K_det_maxinfsize1333.jpg)

## Parameters

- `detect_fn`: object detection inference function
- `max_inference_width` and `max_inference_height`: usually the size your detector will shrink ur oversized image to
- `overlapx_px` and `overlapy_px`: int, overlapping regions
- `pad`: bool, if we add pad to orphan slices.
