from bs4 import BeautifulSoup

def baseline_info_xml(cvat_soup):
    xml_root = "<annotations></annotations>"
    xml_soup = BeautifulSoup(xml_root, 'xml')

    if cvat_soup.find("version"):
        xml_soup.annotations.append(cvat_soup.find("version"))

    if cvat_soup.find("meta"):
        xml_soup.annotations.append(cvat_soup.find("meta"))

    return xml_soup

def f2b_to_xml(all_flatten_dets, smallie_annots, f2b_merged_dicts, xml_soup):
    smallie_annots_filtered = filter_smallie_annots(smallie_annots, f2b_merged_dicts)

    total_annots = 0
    for i, flatten_dets in enumerate(all_flatten_dets):
        xml_img = xml_soup.find("image", {"id": i})
        for flatten_det in flatten_dets:
            xml_info = smallie_annots_filtered[total_annots]

            xml_det = xml_soup.new_tag("box", occluded=xml_info.get("occluded"), source=xml_info.get("source"), z_order=xml_info.get("z_order"))

            for attr in xml_info.find_all("attribute"):
                xml_det.append(attr)

            ltrb, conf, bb_cls = flatten_det
            l, t, r, b = ltrb

            xml_det['label'] = bb_cls
            xml_det['xtl'] = l
            xml_det['ytl'] = t
            xml_det['xbr'] = r
            xml_det['ybr'] = b

            # remove empty attributes
            xml_det.attrs = {k: v for k, v in xml_det.attrs.items() if v is not None}

            xml_img.append(xml_det)

            total_annots += 1

    return xml_soup

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
