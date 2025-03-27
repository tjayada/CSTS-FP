"""
@author: fengyu, wangjun
@date: 20220620
@contact: fengyu_cnyc@163.com
"""

# based on:
# https://github.com/FacePerceiver/facer/blob/main/facer/face_parsing/farl.py
import functools
#import logging.config
#logging.config.fileConfig("config/logging.conf")
#logger = logging.getLogger('sdk')

import torch
import torch.nn.functional as F
import numpy as np
from math import ceil
from itertools import product as product
import torch.backends.cudnn as cudnn

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.transform import *

pretrain_settings = {
    'lapa/448': {
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=1.0),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.8, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.8, warped_shape=(448, 448)),
        'label_names': ['background', 'face', 'rb', 'lb', 're',
                        'le', 'nose',  'ulip', 'imouth', 'llip', 'hair']
    }
}

'''
class FaceParsingModelHandler(BaseModelHandler):
    def __init__(self, model=None, cfg=None, device=None):
        super().__init__(model, cfg, device)
        
        self.model = model.to(self.device)

    def _preprocess(self, image, face_nums):
        """Preprocess the image, such as standardization and other operations.

        Returns:
            A tensor, the shape is 1 x 3 x h x w.
            A dict, {'rects','points','scores','image_ids'} 
        """
        if not isinstance(image, np.ndarray):
            #logger.error('The input should be the ndarray read by cv2!')
            raise ValueError('The input should be the ndarray read by cv2!')
        img = np.float32(image)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).repeat(face_nums, axis=0)
        return torch.from_numpy(img)

    def inference_on_image(self, face_nums: int, images: torch.Tensor, landmarks):
        """Get the inference of the image and process the inference result.

        Returns:
             
        """
        cudnn.benchmark = True
        try:
            image_pre = self._preprocess(images, face_nums)
        except Exception as e:
            raise e
        setting = pretrain_settings['lapa/448']
        images = image_pre.float() / 255.0
        _, _, h, w = images.shape
        simages = images.to(self.device)
        matrix = setting['get_matrix_fn'](landmarks.to(self.device))
        grid = setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
        inv_grid = setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))

        w_images = F.grid_sample(
            simages, grid, mode='bilinear', align_corners=False)

        w_seg_logits, _ = self.model(w_images)  # (b*n) x c x h x w

        seg_logits = F.grid_sample(
            w_seg_logits, inv_grid, mode='bilinear', align_corners=False)
        data_pre = {}
        data_pre['seg'] = {
            'logits': seg_logits,
            'label_names': setting['label_names']
        }
        return data_pre

    def _postprocess(self, loc, conf, scale, input_height, input_width):
        """Postprecess the prediction result.
        Decode detection result, set the confidence threshold and do the NMS
        to keep the appropriate detection box. 

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence), 
            N is the number of detection box.
        """
        pass

'''



"""
@author: fengyu, wangjun
@date: 20220620
@contact: fengyu_cnyc@163.com
"""


class FaceParsingModelHandler(BaseModelHandler):
    def __init__(self, model=None, device=None, cfg=None):
        super().__init__(model, device, cfg)
        self.model = model.to(self.device)

    def _preprocess(self, images, face_nums):
        """Preprocess the batch of images.

        Args:
            images: Tensor of shape (B, C, H, W)
            face_nums: Tensor of shape (B,) containing number of faces per image

        Returns:
            A tensor of shape (total_faces, 3, H, W)
        """
        if not torch.is_tensor(images):
            print('The input should be a PyTorch tensor!')
            raise ValueError()
        
        # Calculate indices for repeating each image according to its face_nums
        batch_indices = torch.arange(len(face_nums), device=images.device)
        repeat_indices = torch.repeat_interleave(batch_indices, face_nums)
        
        # Select and repeat each image according to its face_nums
        return images[repeat_indices]
    
    def inference_on_image(self, face_num: int, image: torch.Tensor, landmarks: torch.Tensor):
        
        image = image.unsqueeze(0)
        face_nums = torch.tensor([face_num], device=image.device)
        return self.inference_on_batch(face_nums, image, landmarks).get('seg')
    

    def inference_on_batch(self, face_nums: torch.Tensor, images: torch.Tensor, landmarks: torch.Tensor):
        """Get the inference of the batched images and process the inference result.
        
        Args:
            face_nums: Tensor of shape (B,) containing number of faces per image
            images: Tensor of shape (B, C, H, W)
            landmarks: Tensor of shape (total_faces, num_landmarks, 2)
        """
        cudnn.benchmark = True
        MAX_BATCH_SIZE = 2
        total_faces = face_nums.sum().item()
        results = []

        try:
            image_pre = self._preprocess(images, face_nums)
        except Exception as e:
            raise e

        for start_idx in range(0, total_faces, MAX_BATCH_SIZE):
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_faces)

            batch_images = image_pre[start_idx:end_idx]
            batch_landmarks = landmarks[start_idx:end_idx]

            setting = pretrain_settings['lapa/448']
            batch_images = batch_images.float() / 255.0
            _, _, h, w = batch_images.shape
            
            matrix = setting['get_matrix_fn'](batch_landmarks)
            grid = setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
            inv_grid = setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))

            w_images = F.grid_sample(
                batch_images, grid, mode='bilinear', align_corners=False)

            w_seg_logits, _ = self.model(w_images)

            seg_logits = F.grid_sample(
                w_seg_logits, inv_grid, mode='bilinear', align_corners=False)
            
            results.append(seg_logits)

            # Clear memory
            del batch_images, batch_landmarks, matrix, grid, inv_grid, w_images, w_seg_logits, seg_logits
            torch.cuda.empty_cache()
        
        final_seg_logits = torch.cat(results, dim=0)
        return {
            'seg': {
                'logits': final_seg_logits,
                'label_names': setting['label_names']
            }
        }

    def _postprocess(self, loc, conf, scale, input_height, input_width):
        """Postprecess the prediction result.
        Decode detection result, set the confidence threshold and do the NMS
        to keep the appropriate detection box. 

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence), 
            N is the number of detection box.
        """
        pass
