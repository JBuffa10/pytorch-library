import pandas as pd
import cv2
import os
import torch
from torch.utils.data import Dataset

class HumanPoseDataset(Dataset):

  def __init__(self, image_path, label_df, img_size, img_transform=None, kp_transform=None):
    self.label_df = label_df
    self.img_path = image_path
    self.img_transform = img_transform
    self.kp_transform = kp_transform
    self.img_size = img_size

  def __len__(self):
    return len(self.label_df)


  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_path = os.path.join(self.img_path,self.label_df['img_name'].iloc[idx])

    image = cv2.imread(img_path)
    h,w = image.shape[:2]

    kp = self.label_df['landmarks'].iloc[idx]

    if self.img_transform and self.kp_transform:
      image = self.img_transform(image)
      kp = kp*[(self.img_size/w),(self.img_size/h)]
      kp = self.kp_transform(image=image, keypoints=kp)


    return image, kp
  
  
