"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
# based on:
# https://github.com/deepinsight/insightface/blob/master/recognition/common/face_align.py

import os
import cv2
import numpy as np
from skimage import transform as trans

from core.image_cropper.BaseImageCropper import BaseImageCropper
from utils.lms_trans import lms106_2_lms5, lms25_2_lms5

src1 = np.array([
     [51.642,50.115],
     [57.617,49.990],
     [35.740,69.007],
     [51.157,89.050],
     [57.025,89.702]], dtype=np.float32)
#<--left 
src2 = np.array([
    [45.031,50.118],
    [65.568,50.872],
    [39.677,68.111],
    [45.177,86.190],
    [64.246,86.758]], dtype=np.float32)

#---frontal
src3 = np.array([
    [39.730,51.138],
    [72.270,51.138],
    [56.000,68.493],
    [42.463,87.010],
    [69.537,87.010]], dtype=np.float32)

#-->right
src4 = np.array([
    [46.845,50.872],
    [67.382,50.118],
    [72.737,68.111],
    [48.167,86.758],
    [67.236,86.190]], dtype=np.float32)

#-->right profile
src5 = np.array([
    [54.796,49.990],
    [60.771,50.115],
    [76.673,69.007],
    [55.388,89.702],
    [61.257,89.050]], dtype=np.float32)

src = np.array([src1,src2,src3,src4,src5])
src_map = {112 : src, 224 : src*2}

arcface_src = np.array([
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041] ], dtype=np.float32 )

arcface_src = np.expand_dims(arcface_src, axis=0)

# In[66]:

# lmk is prediction; src is template
def estimate_norm(lmk, image_size = 112, mode='arcface'):
  assert lmk.shape==(5,2)
  tform = trans.SimilarityTransform()
  lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
  min_M = []
  min_index = []
  min_error = float('inf') 
  if mode=='arcface':
    assert image_size==112
    src = arcface_src
  else:
    src = src_map[image_size]
  for i in np.arange(src.shape[0]):
    tform.estimate(lmk, src[i])
    M = tform.params[0:2,:]
    results = np.dot(M, lmk_tran.T)
    results = results.T
    error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2,axis=1)))
#         print(error)
    if error< min_error:
        min_error = error
        min_M = M
        min_index = i
  return min_M, min_index

def norm_crop(img, landmark, image_size=112, mode='arcface'):
  M, pose_index = estimate_norm(landmark, image_size, mode)
  warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
  return warped

# my class warpper
class FaceRecImageCropper(BaseImageCropper):
    """Implementation of image cropper

    Attributes:
        image: the input image.
        landmarks: using landmarks information to crop.
    """
    def __init__(self):
        super().__init__()
    
    def crop_image_by_mat(self, image, landmarks):
        if len(landmarks) == 106 * 2:
            landmarks = lms106_2_lms5(landmarks)
        if len(landmarks) == 25 * 2:
          landmarks = lms25_2_lms5(landmarks)
        assert(len(landmarks) == 5 * 2)
        landmarks = np.array(landmarks)
        height, width, channel = image.shape
        if channel != 3:
            print('Error input.')
        landmarks = landmarks.reshape((5,2))
        cropped_image = norm_crop(image, landmarks)
        return cropped_image
    
    def reverse_crop_image(self, original_image, cropped_image, landmarks):
      if len(landmarks) == 106 * 2:
        landmarks = lms106_2_lms5(landmarks)
      if len(landmarks) == 25 * 2:
        landmarks = lms25_2_lms5(landmarks)
      assert(len(landmarks) == 5 * 2)
      landmarks = np.array(landmarks).reshape((5, 2))
      
      # Estimate the transformation matrix
      M, pose_index = estimate_norm(landmarks)
      
      # Calculate the inverse transformation matrix
      M_inv = cv2.invertAffineTransform(M)
      
      # Get the dimensions of the original image
      height, width, channel = original_image.shape
      
      # Warp the cropped image back to the original image space
      restored_image = cv2.warpAffine(cropped_image, M_inv, (width, height), borderValue=0.0)
      
      # Create a mask of the cropped image
      mask = np.ones_like(cropped_image, dtype=np.uint8) * 255
      mask = cv2.warpAffine(mask, M_inv, (width, height), borderValue=0.0)
      
      # Combine the restored image with the original image
      combined_image = original_image.copy()
      combined_image[mask > 0] = restored_image[mask > 0]
      
      return combined_image
