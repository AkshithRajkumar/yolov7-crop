import json

tracklets_file_path = "tracklet_bbox.json"

def calculate_iou(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou

with open(tracklets_file_path, "r") as json_file:
    tracklets_dict = json.load(json_file)

first_last_bbox_list = []

for track_id, bbox_list in tracklets_dict.items():
    if bbox_list: 
        first_bbox = bbox_list[0]
        last_bbox = bbox_list[-1]
        first_last_bbox_list.append({"track_id": track_id, "first_bbox": first_bbox, "last_bbox": last_bbox})

# print("First and Last Bbox Values:")
# for entry in first_last_bbox_list:
#     print(f"Track ID: {entry['track_id']}, First Bbox: {entry['first_bbox']}, Last Bbox: {entry['last_bbox']}")


# print("IoU Values:")
for i, entry1 in enumerate(first_last_bbox_list):
    track_id1 = entry1["track_id"]
    first_bbox = entry1["first_bbox"]

    for j, entry2 in enumerate(first_last_bbox_list):
        if i != j:
            track_id2 = entry2["track_id"]
            last_bbox = entry2["last_bbox"]

            iou_value = calculate_iou(first_bbox, last_bbox)

            if iou_value != 0:
                print(f"IoU between Track ID {track_id1} (first bbox) and Track ID {track_id2} (last bbox): {iou_value}")
    
    print("#################################################################")



# Tracklet 1 - Pitcher, 2 - catcher, 3 - referre, 4 - random, 5 - batter, 6 - batter, 9 - catcher
