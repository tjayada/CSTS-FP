"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
#import logging.config
#logging.config.fileConfig("config/logging.conf")
#logger = logging.getLogger('sdk')

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.BuzException import *
from torchvision import transforms

class FaceAlignModelHandler(BaseModelHandler):
    """Implementation of face landmark model handler

    Attributes:
        model: the face landmark model.
        device: use cpu or gpu to process.
        cfg(dict): testing config, inherit from the parent class.
    """
    def __init__(self, model, cfg, device):
        """
        Init FaceLmsModelHandler settings. 
        """
        super().__init__(model, cfg, device)
        self.img_size = self.cfg['img_size']
        
    def inference_on_batch(self, images, batch_dets):
        """Get the inference on a batch of images and process the inference results.
        Args:
            images: torch.Tensor of shape (B, C, H, W) or (B, H, W, C)
            batch_dets: list of length B, where each element contains face detection tensors
        Returns:
            list of torch tensors, each containing landmarks predictions of shape (N, 106, 2)
            where N is the number of faces in each image
        """
        cudnn.benchmark = True
        try:
            # Ensure images are in correct format (B, C, H, W)
            if images.shape[1] != 3:  # If not in CHW format
                images = images.permute(0, 3, 1, 2)
                
            processed_images = []
            batch_metadata = []  # Store xy and boxsize for each detection
            
            for img, dets in zip(images, batch_dets):
                image_processed = []
                for det in dets:
                    processed, metadata = self._preprocess(img, det)
                    image_processed.append(processed)
                    batch_metadata.append(metadata)
                
                processed_images.extend(image_processed)
            
            # Stack all processed images into a single batch
            if not processed_images:
                return []
            
            batch_tensor = torch.stack(processed_images)
            
            # Run inference
            self.model = self.model.to(self.device)
            with torch.no_grad():
                _, landmarks_normal = self.model(batch_tensor)
 
            # Post-process each set of landmarks
            all_landmarks = []
            start_idx = 0
            for dets in batch_dets:
                num_faces = len(dets)
                #print("Number of faces: ", num_faces)

                if num_faces == 0:
                    all_landmarks.append(torch.empty(0, 106, 2, device=self.device))
                    continue
                
                image_landmarks = []
                for i in range(num_faces):
                    metadata = batch_metadata[start_idx + i]
                    landmarks = self._postprocess(landmarks_normal[start_idx + i:start_idx + i + 1], metadata)
                    image_landmarks.append(landmarks)
                                
                all_landmarks.append(torch.stack(image_landmarks) if image_landmarks else torch.empty(0, 106, 2, device=self.device))
                start_idx += num_faces
            #print("\n")
            return all_landmarks
        except Exception as e:
            raise e

    def _preprocess(self, image, det):
        """Preprocess a single image using PyTorch operations.
        Returns:
           A torch tensor, shape: (3, 112, 112).
        """
        if not torch.is_tensor(image):
            raise InputError()
            
        img = image.clone()
        self.image_org = image.clone()
        img = img.float() / 255.0

        # Convert det coordinates to tensor
        xy = torch.tensor([det[0], det[1]], device=self.device)
        zz = torch.tensor([det[2], det[3]], device=self.device)
        wh = zz - xy + 1
        center = (xy + wh / 2).long()
        boxsize = int(torch.max(wh) * 1.2)
        xy = center - boxsize // 2
        self.xy = xy
        self.boxsize = boxsize

        metadata = {'xy': xy, 'boxsize': boxsize}
        
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width = img.shape[1:]
        
        # Calculate padding
        dx = torch.max(torch.tensor(0, device=self.device), -x1)
        dy = torch.max(torch.tensor(0, device=self.device), -y1)
        x1 = torch.max(torch.tensor(0, device=self.device), x1)
        y1 = torch.max(torch.tensor(0, device=self.device), y1)
        edx = torch.max(torch.tensor(0, device=self.device), x2 - width)
        edy = torch.max(torch.tensor(0, device=self.device), y2 - height)
        x2 = torch.min(torch.tensor(width, device=self.device), x2)
        y2 = torch.min(torch.tensor(height, device=self.device), y2)

        # Crop image
        imageT = img[:, y1:y2, x1:x2]

        # Pad if necessary
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            padding = (int(dx), int(edx), int(dy), int(edy))
            imageT = torch.nn.functional.pad(imageT, (padding[0], padding[1], padding[2], padding[3]), mode='constant', value=0)

        # Resize using interpolate
        imageT = imageT.unsqueeze(0)  # Add batch dimension
        imageT = torch.nn.functional.interpolate(imageT, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        imageT = imageT.squeeze(0)  # Remove batch dimension

        return imageT, metadata

    def _postprocess(self, landmarks_normal, metadata=None):
        """Process the predicted landmarks into the form of the original image.
        Returns:
            A torch tensor, the landmarks based on the shape of original image, shape: (106, 2)
        """
        if metadata is not None:
            self.xy = metadata['xy']
            self.boxsize = metadata['boxsize']
        landmarks_normal = landmarks_normal.reshape(landmarks_normal.shape[0], -1, 2)
        landmarks = landmarks_normal[0] * torch.tensor([self.boxsize, self.boxsize], device=self.device) + self.xy

        return landmarks

    # Keep original method for backward compatibility
    def inference_on_image(self, image, dets):
        """Get the inference of the image and process the inference result.
        Returns:
            A torch tensor, the landmarks prediction, shape: (106, 2)
        """
        # Convert single image and detection to batch format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        results = self.inference_on_batch(image, [[dets]])
        return results[0][0]  # Return first landmark set from first image