import argparse
import os
import sys
from pathlib import Path

import torch

from sort_count import *
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode

from PIL import Image
import json
from tqdm import tqdm

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def images_to_video(image_folder, output_path, fps=30):
    image_files = [f for f in os.listdir(image_folder)]
   
    numeric_values = [int(''.join(filter(str.isdigit, filename))) for filename in image_files]
    sorted_indices = sorted(range(len(numeric_values)), key=lambda k: numeric_values[k])
    
    image_files_s = [image_files[i] for i in sorted_indices]

    first_image = cv2.imread(os.path.join(image_folder, image_files_s[0]))
    target_height, target_width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    for image_file in image_files_s:
        # print(image_file)
        image_path = os.path.join(image_folder, image_file)

        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (target_width, target_height))
        
        video_writer.write(frame)

    video_writer.release()

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        trk = False,
):  

    sort_max_age = 30 #max det to wait for tracklet association
    sort_min_hits = 2 #min det to create tracklet   
    sort_iou_thresh = 0.4
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh) 

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    frame_idx = 0
    # max_area = 0

    save_folder = "tracklets_1B/images/"
    # track_ids_file_path = "tracklet_bbox_sort.json"
    # bbox_list = {}
    for path, im, im0s, vid_cap, s in tqdm(dataset):
        frame_idx += 1
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            im_to_save = im0.copy()

            p = Path(p) 
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w

                if trk:
                    dets_to_sort = np.empty((0,6))
                    for x1,y1,x2,y2,conf,detclass in det[:, :6].cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks =sort_tracker.getTrackers()

                    for track in tracks:
                        annotator.draw_trk(line_thickness,track)

                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        annotator.draw_id(bbox_xyxy, identities, categories, names)
                        
                if trk and len(tracked_dets) > 0:
                    for track in tracked_dets: #Looping through tracked detections in the current frame.
                        track_id = int(track[8]) #Extracts the trackingID of the current detection (to identify same obejct across frames).
                        bbox = track[:4].astype(int) #Extracting the bounding box of the detected object.

                        # if track_id in bbox_list:
                        #     bbox_list[track_id].append(bbox.tolist())
                        # else:
                        #     bbox_list[track_id] = [bbox.tolist()]

                        buff = 40
                        cropped_frame = im_to_save[bbox[1]-buff:bbox[3]+buff, bbox[0]-buff:bbox[2]+buff]   
                        if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:

                            bgr_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)

                            save_frame = Image.fromarray(bgr_cropped_frame)

                            tracklet_folder = os.path.join(save_folder, f'{track_id}')
                            os.makedirs(tracklet_folder, exist_ok=True)
                            frame_name = tracklet_folder + f"/{frame_idx}_{track_id}.png"

                            save_frame.save(frame_name)

                        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                        # if area > max_area:
                        #     max_area = area
                        #     max_area_bbox = bbox
    
    for folder_name in os.listdir(save_folder):
        image_folder = os.path.join(save_folder, folder_name)

        output_video_path = os.path.join(save_folder, f"{folder_name}.mp4")
        images_to_video(image_folder, output_video_path)

        print(f"Video saved at: {output_video_path}")

    print("Done!")

    # with open(track_ids_file_path, "w") as json_file:
    #     json.dump(bbox_list, json_file)

    # for track_id in bbox_list.keys():
    #     print(f"Track ID: {track_id}")

    if update:
        strip_optimizer(weights[0])  

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--trk', action='store_true', help='Apply Sort Tracking')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)