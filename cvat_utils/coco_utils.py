def baseline_info(cvat_annots):
    coco_annots = {}
    coco_annots["licenses"] = cvat_annots["licenses"]
    coco_annots["info"] = cvat_annots["info"]
    coco_annots["categories"] = cvat_annots["categories"]
    return coco_annots

def f2b_to_coco(all_flatten_dets, categories):
    coco_dets = []
    total_annots = 0
    for i, flatten_dets in enumerate(all_flatten_dets):
        for flatten_det in flatten_dets:
            # coco_det = {"image_id": i+1}
            coco_det = {"image_id": i+1, "segmentation": [], "iscrowd": 0}

            ltrb, conf, bb_cls = flatten_det

            total_annots += 1
            coco_det["id"] = total_annots

            cat_id = next((item for item in categories if item["name"] == bb_cls), None)['id']
            coco_det["category_id"] = cat_id

            l, t, r, b = ltrb
            w = r - l
            h = b - t
            area = w * h
            coco_det["bbox"] = [l, t, w, h]
            coco_det["area"] = area

            coco_dets.append(coco_det)

    return coco_dets
