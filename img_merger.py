import cv2
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup

from f2b import F2B, draw
from cvat_utils.coco_utils import baseline_info_coco, f2b_to_coco
from cvat_utils.xml_utils import baseline_info_xml, f2b_to_xml


parser = argparse.ArgumentParser()
parser.add_argument('--f2b_file', help='path to file for f2b settings', default='cvat_images/originals/f2b_set.json', type=str)
parser.add_argument('--cvat_annot_file', help='path to file for cvat annotations', default='cvat_images/originals/cvat_annot_coco.json', type=str)
parser.add_argument('--biggie_annot_file', help='path to output file for biggie annotations', default='cvat_images/originals/biggie_annot_coco.json', type=str)
parser.add_argument('--vis_extension', help='extension for visualized smallies and annotated image output', default='jpg', type=str)
args = parser.parse_args()

f2b_filepath = Path(args.f2b_file)
assert f2b_filepath.is_file(), 'img data file doesnt exist'
with open(str(f2b_filepath)) as json_file:
    f2b_data = json.load(json_file)

biggie_annots_path = Path(args.biggie_annot_file)

smol_annots_dir = f2b_filepath.parent / 'smol_annots'
Path.mkdir(smol_annots_dir, exist_ok=True)

cvat_annots_path = Path(args.cvat_annot_file)
assert cvat_annots_path.is_file(), 'img data file doesnt exist'

# check if 'coco' or 'cvat for images' format
if cvat_annots_path.suffix == '.json':
    assert biggie_annots_path.suffix=='.json', 'biggie_annot_file needs to be a json'

    with open(str(cvat_annots_path)) as json_file:
        cvat_annots = json.load(json_file)

    biggie_annots = baseline_info_coco(cvat_annots)

    # get category labels
    cats_f2b = {}
    for cat_dict in cvat_annots['categories']:
        cats_f2b[cat_dict['id']] = cat_dict['name']

    biggie_images = []
    all_flatten_dets = []
    all_smallie_annots_coco = []
    f2b_merged_dicts = []
    for i, img_name in enumerate(f2b_data):
        # read original image and f2b settings
        f2b_settings = f2b_data[img_name]

        input_path = Path(f2b_settings["img_path"])
        biggie = cv2.imread(str(input_path))

        if biggie is None:
            print(f'{input_path} does not exist')
            continue

        biggie_images.append({"id": i+1, "width": biggie.shape[1], "height": biggie.shape[0], "file_name": img_name, "license": 0})

        f2b = F2B(
            max_inference_width = f2b_settings['max_inference_width'],
            max_inference_height = f2b_settings['max_inference_height'],
            overlapx_px = f2b_settings['overlapx_px'],
            overlapy_px = f2b_settings['overlapy_px'],
            )
        num_smols = f2b.register(biggie.shape[:2])

        # get all annotations of all smallies for the image
        img_annots = []
        img_smallie_annots_coco = []
        for smallie_path in f2b_settings["smols"]:
            # find image_id for smallie
            smallie_name = Path(smallie_path).name
            image_data = next((item for item in cvat_annots['images'] if item["file_name"] == smallie_name), None)

            if image_data is not None:
                # get all annotations for smallie
                smallie_annots_coco = [element for element in cvat_annots['annotations'] if element['image_id'] == image_data['id']]
                # print(smallie_annots_coco)
                img_smallie_annots_coco.append(smallie_annots_coco)

                smallie_annots_f2b = []
                # convert coco annotations into detect_get_box_in format
                for smallie_annot in smallie_annots_coco:
                    l, t, w, h = smallie_annot['bbox']
                    r = l + w
                    b = t + h
                    conf = 1
                    bb_cls = cats_f2b[smallie_annot['category_id']]

                    smallie_annot_f2b = ([l,t,r,b], conf, bb_cls)
                    smallie_annots_f2b.append(smallie_annot_f2b)

                # sanity check
                smol = cv2.imread(str(smallie_path)).copy()
                draw.draw_dets(smol, smallie_annots_f2b)
                out_path = smol_annots_dir / f'{Path(smallie_path).stem}.{args.vis_extension}'
                cv2.imwrite(str(out_path), smol)

                img_annots.append(smallie_annots_f2b)
            else:
                img_annots.append([])
                img_smallie_annots_coco.append([])
                print(f'annotations for {smallie_name} not found')
        # print(img_annots)
        all_smallie_annots_coco.append(img_smallie_annots_coco)

        # map annotation results back to original image
        if num_smols > 0:
            flatten_dets, smol_indices = f2b.deconflicting_union(img_annots)
        else:
            flatten_dets, smol_indices = None, None
        all_flatten_dets.append(flatten_dets)
        f2b_merged_dicts.append(f2b.merged_dict)

        biggie_show = biggie.copy()
        draw.draw_biggie(biggie_show, flatten_dets, f2b.smol_coords, smol_indices)

        out_path = smol_annots_dir / f'{input_path.stem}_annot.{args.vis_extension}'
        cv2.imwrite(str(out_path), biggie_show)

    # create annotation json for biggies
    biggie_annots["images"] = biggie_images
    biggie_dets = f2b_to_coco(all_flatten_dets, cvat_annots['categories'], all_smallie_annots_coco, f2b_merged_dicts)

    biggie_annots["annotations"] = biggie_dets

    with open(str(biggie_annots_path), 'w+') as outfile:
        outfile.write(json.dumps(biggie_annots, indent=4))
        # json.dump(biggie_annots, outfile)

elif cvat_annots_path.suffix == '.xml':
    assert biggie_annots_path.suffix=='.xml', 'biggie_annot_file needs to be a xml'
    with open(str(cvat_annots_path), 'r') as xml_file:
        cvat_contents = xml_file.read()
        cvat_soup = BeautifulSoup(cvat_contents, 'xml')

    xml_soup = baseline_info_xml(cvat_soup)

    all_flatten_dets = []
    all_smallie_annots_xml = []
    f2b_merged_dicts = []
    for i, img_name in enumerate(f2b_data):
        # read original image and f2b settings
        f2b_settings = f2b_data[img_name]

        input_path = Path(f2b_settings["img_path"])
        biggie = cv2.imread(str(input_path))

        if biggie is None:
            print(f'{input_path} does not exist')
            continue

        xml_img = xml_soup.new_tag("image", id=i, width=biggie.shape[1], height=biggie.shape[0])
        xml_img['name'] = img_name
        xml_soup.annotations.append(xml_img)

        f2b = F2B(
            max_inference_width = f2b_settings['max_inference_width'],
            max_inference_height = f2b_settings['max_inference_height'],
            overlapx_px = f2b_settings['overlapx_px'],
            overlapy_px = f2b_settings['overlapy_px'],
            )
        num_smols = f2b.register(biggie.shape[:2])

        # get all annotations of all smallies for the image
        img_annots = []
        img_smallie_annots_xml = []
        for smallie_path in f2b_settings["smols"]:
            # find image_id for smallie
            smallie_name = Path(smallie_path).name
            image_data = cvat_soup.find("image", {"name": smallie_name})
            # print(image_data)

            if image_data is not None:
                smallie_annots_xml = [annot_box for annot_box in image_data.find_all("box")]
                # print(smallie_annots_xml)
                img_smallie_annots_xml.append(smallie_annots_xml)

                smallie_annots_f2b = []
                # convert xml annotations into detect_get_box_in format
                for smallie_annot in smallie_annots_xml:
                    t = float(smallie_annot.get("ytl"))
                    l = float(smallie_annot.get("xtl"))
                    b = float(smallie_annot.get("ybr"))
                    r = float(smallie_annot.get("xbr"))
                    conf = 1
                    bb_cls = smallie_annot.get("label")

                    smallie_annot_f2b = ([l,t,r,b], conf, bb_cls)
                    smallie_annots_f2b.append(smallie_annot_f2b)
                # print(smallie_annots_f2b)

                # sanity check
                smol = cv2.imread(str(smallie_path)).copy()
                draw.draw_dets(smol, smallie_annots_f2b)
                out_path = smol_annots_dir / f'{Path(smallie_path).stem}.{args.vis_extension}'
                cv2.imwrite(str(out_path), smol)

                img_annots.append(smallie_annots_f2b)
            else:
                img_annots.append([])
                img_smallie_annots_xml.append([])
                print(f'annotations for {smallie_name} not found')
        # print(img_annots)
        all_smallie_annots_xml.append(img_smallie_annots_xml)

        # map annotation results back to original image
        if num_smols > 0:
            flatten_dets, smol_indices = f2b.deconflicting_union(img_annots)
        else:
            flatten_dets, smol_indices = None, None
        all_flatten_dets.append(flatten_dets)
        f2b_merged_dicts.append(f2b.merged_dict)

        biggie_show = biggie.copy()
        draw.draw_biggie(biggie_show, flatten_dets, f2b.smol_coords, smol_indices)

        out_path = smol_annots_dir / f'{input_path.stem}_annot.{args.vis_extension}'
        cv2.imwrite(str(out_path), biggie_show)

    # create annotation xml for biggies
    xml_soup = f2b_to_xml(all_flatten_dets, all_smallie_annots_xml, f2b_merged_dicts, xml_soup)

    with open(str(biggie_annots_path), 'w+') as outfile:
        outfile.write(xml_soup.prettify())
