"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com 
"""

#import logging.config
#logging.config.fileConfig("config/logging.conf")
#logger = logging.getLogger('sdk')

import torch
import numpy as np
from math import ceil
from itertools import product as product
import torch.backends.cudnn as cudnn

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.BuzException import *


class FaceDetModelHandler(BaseModelHandler):
    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        self.variance = torch.tensor(list(self.cfg['variance'])).to(device)
        self.model = self.model.to(device)
        self.mean = torch.tensor([104, 117, 123]).view(1, 3, 1, 1).to(device)

    def inference_on_batch(self, images):
        """Get the inference of a batch of images.
        
        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 255]
        
        Returns:
            List of N * (x, y, w, h, confidence) arrays, one per image
        """
        cudnn.benchmark = True
        
        if not isinstance(images, torch.Tensor):
            raise InputError('Input must be a PyTorch tensor')
        
        if images.dim() != 4:
            raise InputError(f'Expected 4D tensor (B,C,H,W), got {images.dim()}D')
        
        # Get original image sizes before any processing
        image_sizes = [(h, w) for h, w in zip(
            images.shape[2] * torch.ones(images.shape[0], dtype=torch.int),
            images.shape[3] * torch.ones(images.shape[0], dtype=torch.int)
        )]
        
        scales = torch.tensor([[w, h, w, h] for h, w in image_sizes]).to(self.device)
        
        # Preprocess batch
        images = images.to(self.device)
        images = images - self.mean  # Broadcasting will handle this efficiently

        with torch.no_grad():
            loc, conf, landms = self.model(images)
       
        # Process each image's detections
        all_dets = []
        for i in range(len(image_sizes)):
            dets = self._postprocess(
                loc[i:i+1], 
                conf[i:i+1], 
                scales[i], 
                image_sizes[i][0],  # height
                image_sizes[i][1]   # width
            )
            all_dets.append(dets)
           
        return all_dets

    def inference_on_image(self, image):
        """Legacy single-image inference method.
        
        Args:
            image: Tensor of shape (C, H, W) in range [0, 255]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        dets = self.inference_on_batch(image)[0]
        return dets

    def _postprocess(self, loc, conf, scale, input_height, input_width):
        """Postprocess the prediction result.
        All inputs are expected to be PyTorch tensors on the correct device.
        """
        priorbox = PriorBox(self.cfg, image_size=(input_height, input_width))
        priors = priorbox.forward().to(self.device)
        
        boxes = self.decode(loc.squeeze(0), priors, self.variance)
        boxes = boxes * scale
        
        scores = conf.squeeze(0)[:, 1]
        
        # Move NMS operations to GPU
        mask = scores > self.cfg['confidence_threshold']
        boxes = boxes[mask]
        scores = scores[mask]
   
        # Sort on GPU
        scores, order = scores.sort(descending=True)
        boxes = boxes[order]

        # Move NMS operations to GPU
        mask = scores > self.cfg['confidence_threshold']
        boxes = boxes[mask]
        scores = scores[mask]

        # Prepare dets tensor for NMS (combine boxes and scores)
        dets = torch.cat((boxes, scores.unsqueeze(1)), dim=1)

        # Do NMS on GPU
        nms_threshold = 0.2
        keep = self.gpu_nms(dets, nms_threshold)

        if len(keep) == 0:
            return None

        # Only move final results to CPU
        dets = dets[keep]
        return dets
    
    # Adapted from https://github.com/chainer/chainercv
    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes

        Return:
            decoded bounding box predictions
        """
        boxes = torch.cat((priors[:, :2], priors[:, 2:]), 1)
        boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
        boxes[:, 2:] = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def gpu_nms(self, dets, thresh):
        """GPU version of NMS.
        
        Args:
            dets: tensor of shape (N, 5) where each row is [x1, y1, x2, y2, score]
            thresh: IoU threshold for NMS
        
        Returns:
            The kept indices after NMS.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        _, order = scores.sort(descending=True)
        keep = []

        count = 0
        while order.size(0) > 0:
            
            count += 1

            i = order[0]
            keep.append(i)
            
            if order.size(0) == 1:
                break
                
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1 + 1).clamp(min=0)
            h = (yy2 - yy1 + 1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= thresh).nonzero().squeeze()
            
            if ids.numel() == 0:
                break
            order = order[ids + 1]

            if order.numel() == 0:
                break
                
            if order.dim() == 0:
                order = order.unsqueeze(0)
            
        return torch.tensor(keep, device=dets.device)






# Adapted from https://github.com/biubug6/Pytorch_Retinafacey
class PriorBox(object):
    """Compute the suitable parameters of anchors for later decode operation

    Attributes:
        cfg(dict): testing config.
        image_size(tuple): the input image size.
    """

    def __init__(self, cfg, image_size=None):
        """
        Init priorBox settings related to the generation of anchors. 
        """
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        return output
