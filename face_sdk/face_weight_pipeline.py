
import os
import sys

# class gets called from slowfast/models/custom_multimodal_builder.py
# so need to add the path to the sys.path

sys.path.append('face_sdk/.')

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from utils.draw import draw_bchw
from utils.show import save_bchw
from utils.show import get_bchw 
from core.model_loader.face_parsing.FaceParsingModelLoader import FaceParsingModelLoader
from core.model_handler.face_parsing.FaceParsingModelHandler import FaceParsingModelHandler
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

from slowfast.datasets import utils

import warnings
warnings.filterwarnings("ignore")

from einops import rearrange

from torch.profiler import profile, record_function, ProfilerActivity

# new implementation with RGB to BGR 
class ImagePreprocessor(torch.nn.Module):
    """Preprocesses images for face detection model.
    
    Performs:
    1. RGB to BGR conversion
    2. Min-max normalization
    3. Scaling to [0, 255]
    """
    def __init__(self):
        super().__init__()
        self.normalizer = ImageNormalizer(to_255=True)
        # Define RGB->BGR conversion index once
        self.rgb_to_bgr_idx = torch.tensor([2, 1, 0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) in RGB format
            
        Returns:
            Preprocessed tensor in BGR format, normalized to [0, 255]
        """
        # Move index to correct device
        if self.rgb_to_bgr_idx.device != images.device:
            self.rgb_to_bgr_idx = self.rgb_to_bgr_idx.to(images.device)
            
        # Convert to float if needed
        images = images.float()
        
        # Convert RGB to BGR
        images = images[:, self.rgb_to_bgr_idx]
        
        # Normalize
        images = self.normalizer(images)
        
        return images


class ImageNormalizer(torch.nn.Module):
    """Module to normalize images to [0, 255] range using min-max normalization."""
    
    def __init__(self, to_255: bool = True):
        super().__init__()
        self.to_255 = to_255
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        images_flat = images.view(B, C, -1)
        
        min_vals = images_flat.min(dim=2, keepdim=True)[0]
        max_vals = images_flat.max(dim=2, keepdim=True)[0]
        
        eps = torch.finfo(images.dtype).eps
        scale = torch.clamp(max_vals - min_vals, min=eps)
        
        normalized = (images_flat - min_vals) / scale
        normalized = normalized.view(B, C, H, W)
        
        if self.to_255:
            normalized = normalized * 255
            
        return normalized

def check_image_stats(images, stage=""):
    print(f"\nImage stats at {stage}:")
    print(f"Range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mean: {images.mean():.3f}")
    print(f"Std: {images.std():.3f}")


class ImagePostprocessor(torch.nn.Module):
    """Postprocesses images to revert preprocessing steps.
    
    Performs:
    1. BGR to RGB conversion
    2. Reverses min-max normalization
    3. Scaling back to original mean and std
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.normalizer = ImageNormalizer()
        self.mean = mean
        self.std = std
        # Define BGR->RGB conversion index once
        self.bgr_to_rgb_idx = torch.tensor([2, 1, 0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) in BGR format
            
        Returns:
            Postprocessed tensor in RGB format, with original mean and std
        """
        # Move index to correct device
        if self.bgr_to_rgb_idx.device != images.device:
            self.bgr_to_rgb_idx = self.bgr_to_rgb_idx.to(images.device)
            
        # Convert to float if needed
        images = images.float()
        
        # Convert BGR to RGB
        images = images[:, self.bgr_to_rgb_idx]
        
        # Reverse normalization
        if self.normalizer.to_255:
            images = images / 255
        
        B, C, H, W = images.shape
        images_flat = images.view(B, C, -1)
        
        min_vals = images_flat.min(dim=2, keepdim=True)[0]
        max_vals = images_flat.max(dim=2, keepdim=True)[0]
        
        eps = torch.finfo(images.dtype).eps
        scale = torch.clamp(max_vals - min_vals, min=eps)
        
        denormalized = images_flat * scale + min_vals
        denormalized = denormalized.view(B, C, H, W)
        
        # Scale back to original mean and std
        denormalized = rearrange(denormalized, 'b c h w -> b h w c')
        denormalized = utils.tensor_normalize(denormalized, self.mean, self.std)
        denormalized = rearrange(denormalized, 'b h w c -> b c h w')
        
        return denormalized

class faceParsingPipeline(nn.Module):
    def __init__(self, model_path: str, device, cfg):
        super(faceParsingPipeline, self).__init__()

        # test if this works for multiple GPUs and parallel processing
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.mean = torch.tensor(cfg.DATA.MEAN, device=self.device)
        self.std = torch.tensor(cfg.DATA.STD, device=self.device)

        self.preprocessor = ImagePreprocessor().to(self.device)
        #self.mean = torch.tensor([0.45, 0.45, 0.45], device=self.device)
        #self.std = torch.tensor([0.225, 0.225, 0.225], device=self.device)
        self.postprocessor = ImagePostprocessor(self.mean, self.std).to(self.device)


        scene = 'non-mask' # scene is non-mask for every model
        
        model_category = 'face_detection'
        model_name = 'face_detection_1.0'
        self.faceDetModelHandler = self.get_faceDetModelHandler(model_path, model_category, model_name)
        
        model_category = 'face_alignment'
        model_name = 'face_alignment_1.0'
        self.faceAlignModelHandler = self.get_faceAlignModelHandler(model_path, model_category, model_name)
        
        model_category = 'face_parsing'
        model_name = 'face_parsing_1.0'
        self.faceParsingModelHandler = self.get_faceParsingModelHandler(model_path, model_category, model_name)
        
        self.face_cropper = FaceRecImageCropper()

    def get_faceDetModelHandler(self, model_path: str, model_category: str, model_name:str) -> FaceDetModelHandler:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, cfg, self.device)
        return faceDetModelHandler

    def get_faceAlignModelHandler(self, model_path: str, model_category: str, model_name: str) -> FaceAlignModelHandler:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, cfg, self.device)
        return faceAlignModelHandler

    def get_faceParsingModelHandler(self, model_path: str, model_category: str, model_name: str) -> FaceParsingModelHandler:
        faceParsingModelLoader = FaceParsingModelLoader(model_path, model_category, model_name)
        model, cfg = faceParsingModelLoader.load_model()
        faceParsingModelHandler = FaceParsingModelHandler(model, cfg, self.device)
        return faceParsingModelHandler
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        # input shape :  (8, 3, 8, 256, 256)

        with torch.no_grad():  # no need to compute gradients for inference
            
            v = rearrange(input_tensor, 'b c t h w -> b t c h w')

            for sample in range(v.shape[0]):

                # Clear cache before processing each sample
                torch.cuda.empty_cache()
                
                #check_image_stats(v[sample], stage="before_preprocess")

                # Preprocess the batch
                batch_preprocessed = self.preprocessor(v[sample])

                #check_image_stats(batch_preprocessed, stage="after_preprocess")

                # Detect faces
                dets = self.faceDetModelHandler.inference_on_batch(batch_preprocessed)

                # If no faces were detected, skip the rest of the processing
                faces_detected_at_indices = [i for i, x in enumerate(dets) if x is not None]
                if not faces_detected_at_indices:
                    continue
                    
                # Only process frames with detected faces
                batch_with_faces = batch_preprocessed[faces_detected_at_indices]
                dets_with_faces = [dets[i] for i in faces_detected_at_indices]

                # Process Landmarks
                landmarks = self.faceAlignModelHandler.inference_on_batch(batch_with_faces, dets_with_faces)
                landmarks_list = [e[[104, 105, 54, 84, 90]] for elem in landmarks for e in elem]
                batch_landmarks = torch.stack(landmarks_list, dim=0)

                # Get the number of faces detected in each frame
                batch_face_nums = torch.tensor([len(x) for x in dets_with_faces], device=batch_with_faces.device)

                # Parse faces
                faces = self.faceParsingModelHandler.inference_on_batch(batch_face_nums, batch_with_faces, batch_landmarks)  # parse detected and aligned faces
                
                # Draw the parsed images
                parsed_images = draw_bchw(batch_with_faces, faces, batch_face_nums)

                # Get parsed images back to the original mean and std
                parsed_images = self.postprocessor(parsed_images)

                # Update the original batch with the parsed images
                v[sample][faces_detected_at_indices] = parsed_images

                #check_image_stats(v[sample], stage="after_post_processing")

                # Clear cache after processing each sample
                del batch_preprocessed, dets, batch_with_faces, landmarks
                del batch_landmarks, faces, parsed_images
                torch.cuda.empty_cache()

                
        v = rearrange(v, 'b t c h w -> b c t h w')
        return v