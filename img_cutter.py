import cv2
import json
import argparse
from pathlib import Path

from f2b import F2B, draw


parser = argparse.ArgumentParser()
parser.add_argument('--img', help='path of image file', default='examples/seoul-station_4K.png', type=str)
parser.add_argument('--out_dir', help='path of directory to store smol images for cvat annotation', default='cvat_images', type=str)
parser.add_argument('--f2b_width', help='f2b max_inference_width', default=1333, type=int)
parser.add_argument('--f2b_height', help='f2b max_inference_height', default=1333, type=int)
parser.add_argument('--f2b_x', help='f2b overlapx_px', default=100, type=int)
parser.add_argument('--f2b_y', help='f2b overlapy_px', default=100, type=int)
parser.add_argument('--out_f2b', help='path to output file for f2b settings', default='cvat_images/originals/f2b_set.json', type=str)
parser.add_argument('--smol_extension', help='extension for smol image output', default='jpg', type=str)
args = parser.parse_args()

input_image = args.img
input_path = Path(input_image)
assert input_path.is_file(), 'img not found'
biggie = cv2.imread(input_image)
print(biggie.shape)

out_dir = Path(args.out_dir)
Path.mkdir(out_dir, exist_ok=True)

# save original image
out_path = out_dir / 'originals' / f'{input_path.name}'
cv2.imwrite(str(out_path), biggie)

# f2b settings
max_inference_width = args.f2b_width
max_inference_height = args.f2b_height
overlapx_px = args.f2b_x
overlapy_px = args.f2b_y

img_data = {input_path.name: {'max_inference_width': max_inference_width, 'max_inference_height': max_inference_height, 'overlapx_px': overlapx_px, 'overlapy_px': overlapy_px, 'img_path': str(out_path)}}

f2b = F2B(
        max_inference_width = max_inference_width,
        max_inference_height = max_inference_height,
        overlapx_px = overlapx_px,
        overlapy_px = overlapy_px,
        )

num_smols = f2b.register(biggie.shape[:2])

smol_frames = f2b.slice_n_dice(biggie)

biggie_show = biggie.copy()
draw.draw_smol(biggie_show, f2b.smol_coords)
out_path = out_dir / 'originals' / 'smol_annots' / f'{input_path.stem}_smols.png'
cv2.imwrite(str(out_path), biggie_show)

smol_paths = []
for i, smallie in enumerate(smol_frames):
    out_path = out_dir / f'{input_path.stem}_{i}.{args.smol_extension}'
    smol_paths.append(str(out_path))
    cv2.imwrite(str(out_path), smallie)
img_data[input_path.name]['smols'] = smol_paths

json_path = Path(args.out_f2b)
if json_path.is_file():
    with open(str(json_path)) as outfile:
        data = json.load(outfile)
        data.update(img_data)

    with open(str(json_path), 'w+') as outfile:
        outfile.write(json.dumps(data, indent=4))
else:
    with open(str(json_path), 'w+') as outfile:
        outfile.write(json.dumps(img_data, indent=4))