def baseline_info_coco(cvat_annots):
    coco_annots = {}
    coco_annots["licenses"] = cvat_annots["licenses"]
    coco_annots["info"] = cvat_annots["info"]
    coco_annots["categories"] = cvat_annots["categories"]
    return coco_annots

def f2b_to_coco(all_flatten_dets, categories, smallie_annots, f2b_merged_dicts):
    smallie_annots_filtered = filter_smallie_annots(smallie_annots, f2b_merged_dicts)

    coco_dets = []
    total_annots = 0
    for i, flatten_dets in enumerate(all_flatten_dets):
        for flatten_det in flatten_dets:
            coco_det = {"image_id": i+1}

            coco_info = smallie_annots_filtered[total_annots]
            info = dict((key, coco_info[key]) for key in ['segmentation', 'iscrowd', 'attributes'] if key in coco_info)
            coco_det.update(info)

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

def filter_smallie_annots(smallie_annots, f2b_merged_dicts):
    starting_indices = []
    total_annots = 0
    smallie_annots_flatten = []
    for smallie_annot_img in smallie_annots:
        starting_indices_img = []
        for smallie_annot_smol in smallie_annot_img:
            starting_indices_img.append(total_annots)
            total_annots += len(smallie_annot_smol)
            for smallie_annot in smallie_annot_smol:
                smallie_annots_flatten.append(smallie_annot)
        starting_indices.append(starting_indices_img)
    # print(starting_indices)
    # print(f'num annots: {len(smallie_annots_flatten)}')

    merge_indices = []
    for i, merged_img in enumerate(f2b_merged_dicts):
        for merged in merged_img:
            idx = starting_indices[i][merged[0]] + merged[1]
            merge_indices.append(idx)

    merge_indices.sort(reverse=True)
    # print(merge_indices)

    # remove merged coco annotations (start from behind because del by index)
    for idx in merge_indices:
        del smallie_annots_flatten[idx]
    # print(f'annots left: {len(smallie_annots_flatten)}')

    return smallie_annots_flatten
