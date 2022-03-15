"""Run inference with a YOLOv5 model on images 

Usage:
    
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

import cv2

import torch
import torch.backends.cudnn as cudnn

#sys.path.insert(1, '/home/mj/AAA/yolo5/')

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.dirname(parentdir))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox



class yolo5_detector():
    def __init__(self, weight_path, imgsz):

        self.weights=weight_path       # model.pt path(s)
        self.imgsz=imgsz               # inference size (pixels)
        self.conf_thres=0.35         # confidence threshold
        self.iou_thres= 0.45#0.45          # NMS IOU threshold
        self.max_det=1000            # maximum detections per image
        self.device='cuda:0'         # cuda device i.e. 0 or 0123 or cpu
        self.classes=None            # filter by class: --class 0 or --class 0 2 3
        self.agnostic_nms=False      # class-agnostic NMS
        self.augment=False           # augmented inference
        self.visualize=False         # visualize features
        self.half=False              # use FP16 half-precision inference


        device = select_device(self.device)
        # self.half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load( self.weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # self.model stride
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        # names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names


        if self.half:
            self.model.half()  # to FP16

        cudnn.benchmark = True


    def predict(self, img):
    
        # img = cv2.resize( img, (1280,736) )
        # img = cv2.resize( img, (640,384) )

        # img0 = img

        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        
        # pred = self.model(img,augment=self.augment,visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False)[0]
        pred = self.model(img,augment=self.augment, visualize=self.visualize)[0]

        # Apply NMS 
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det) 
        

        return pred

