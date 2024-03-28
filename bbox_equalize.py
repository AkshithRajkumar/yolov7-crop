import torch
from tqdm import tqdm
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams

save_folder = "tracklets_1B/images/"

import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from PIL import Image

def bbox_or(bbox, source):

    print(bbox)

    device = select_device(0)
    weights=ROOT / 'yolov5s-seg.pt'
    data = ROOT / 'data/coco128.yaml'
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride_n, _, pt_n = model.stride, model.names, model.pt
    imgsz_n = check_img_size((640, 640), s=stride_n)  # check image size

    dataset_n = LoadImages(source, img_size=imgsz_n, stride=stride_n, auto=pt_n)

    for path, im, im0s, _, s in dataset_n:

        cropped_frame = im0s[bbox[1]:bbox[3], bbox[0]:bbox[2]]   

        # if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
        save_frame = Image.fromarray(cropped_frame)

        tracklet_folder = os.path.join(save_folder, f'{track_id}')
        os.makedirs(tracklet_folder, exist_ok=True)
        frame_name = tracklet_folder + f"/{frame_idx}_{track_id}.png"

        save_frame.save(frame_name)


if __name__ == '__main__':

    source = "1B.mp4"
    bbox_or([399,196,903,659], source)
    print("Done!")