## --------------------------------
## LOAD LIBRARIES
## ------------------------------------

import os
import requests
import cv2
import scipy.io
import numpy as np 
import pandas as pd
import torch
from torchsummary import summary
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from human_pose.utils.HumanPoseDataset import HumanPoseDataset
from human_pose.utils.HeatMap import HeatMap
from human_pose.utils.load_mpii_data import generate_dataset_obj


## ----------------------------------------
## ASSIGN DATA DIRECTORIES
## ----------------------------------------

root_dir = "/content/drive/MyDrive/Colab Notebooks/HumanPoseProject"
img_path = "/content/drive/MyDrive/Colab Notebooks/HumanPoseProject/mpii_human_pose_images/images"
label_path = "/content/drive/MyDrive/Colab Notebooks/HumanPoseProject/mpii_human_pose_images/Annotations/mpii_human_pose_v1_u12_1.mat"



## --------------------------------------
## LOAD MPii LABELS
## ---------------------------------------

mat = scipy.io.loadmat(label_path, struct_as_record=False)['RELEASE']
dataset_obj = generate_dataset_obj(mat)
dataset = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dataset_obj.items() ]))

# Create and fill image name column
dataset['img_name'] = ''
for idx, row in dataset['annolist'].items():
    dataset.at[idx, 'img_name'] = row['image']['name']

# Split dataframes into training/testing & drop uneeded columns
train = dataset.loc[dataset['img_train'] == 1].reset_index().drop(columns=['index', 'version', 'img_train', 'video_list'])


# Create column to holds POIs (joint coordinates) for training
train['points'] = ''
for idx, row in train['annolist'].items():
    length = len(row['annorect'])
    points = []
    for i in range(length):
        try:
            points.append(row['annorect'][i]['annopoints']['point'])
        except:
            continue
    train.at[idx, 'points'] = np.array(points, dtype='object').reshape(-1)

train.dropna(axis=0, subset=['img_name', 'points'], inplace=True)


# Refine ROIs to multidimensional arrays of x, y coordinates for each joint in new landmarks column
train['landmarks'] = ''
train['ids'] =''
for idx, row in train['points'].items():
    length = len(row)
    coords = []
    ids = []
    for i in range(length):
        try:
            coords.append([row[i]['x'], row[i]['y']])
            ids.append([row[i]['id']])
        except:
            continue
    train.at[idx, 'landmarks'] = np.array(coords)
    train.at[idx, 'ids'] = np.array(ids)

train['acts'] = ''
for idx, row in train['act'].items():
    train.at[idx, 'acts'] = row['act_id']

# Drop empty arrays
for idx, row in train['landmarks'].items():
    if row.size == 0:
        train.at[idx, 'landmarks'] = np.nan

# Clean up dataframes
train.drop(columns=['annolist', 'points'], inplace=True)
train.dropna(axis=0, subset=['img_name', 'landmarks'], inplace=True)

# Only needed if training on specific number of key points
df = train.copy()
df['n_kp'] = df['landmarks'].str.len()
df_16 = df[df['n_kp']==16]




## ---------------------------------------
## DATA TRANSFORMATIONS
## ---------------------------------------

img_tranforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((256,256)),
        transforms.ToTensor()
])
kp_transforms = HeatMap(sigma=4)

transformed_data = HumanPoseDataset(
    image_path = img_path, 
    label_df = df_16, 
    img_size=256,
    img_transform=img_tranforms,
    kp_transform=kp_transforms
    )



## -------------------------------------
## DATALOADERS
## --------------------------------------

lengths = [int(len(transformed_data)*.7), int(len(transformed_data)*.2), int(len(transformed_data)*.1)+2]
train_, val_, test_ = random_split(transformed_data, lengths=lengths)
test_loader = DataLoader(test_, batch_size=4, num_workers=2)
train_loader = DataLoader(train_, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_, batch_size=4, shuffle=False, num_workers=2)



## ----------------------------------------
## LIGHTNING WORKFLOW
## ---------------------------------------

class Model(pl.LightningModule):
    def __init__(
        self,
        n_channels = 1, 
        n_kp = 16,
        lr = 1e-3
        ):
        super().__init__()
        self.save_hyperparameters()
        self.n_channels = n_channels
        self.n_kp = n_kp
        self.lr = lr
        self.loss = nn.MSELoss()

        self.model = Unet()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y = torch.flatten(y, start_dim=2)
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        loss = self.loss(out,y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y = torch.flatten(y, start_dim=2)
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        loss = self.loss(out,y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x,y = batch
        y = torch.flatten(y, start_dim=2)
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        loss = self.loss(out,y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x,y = batch
        pred = self.model(x)
        return pred

    # def validation_epoch_end(self, outputs):
    #     loss_val = torch.tensor([x['val_loss'] for x in outputs]).mean() # Switch to torch.tensor()
    #     log_dict = {'val_loss' : loss_val}
    #     return {'log' : log_dict, 'val_loss' : log_dict['val_loss'], 'progress_bar' : log_dict}

    def configure_optimizers(self):
      opt = torch.optim.Adam(self.parameters(), lr = self.lr)
      # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
      return opt
    
    
model = Model()


summary(model, input_size=(1,256,256),batch_size=4)



## ------------------------------------
## LIGHTNING TRAINER
## -----------------------------------

tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = Trainer(logger=tb_logger, progress_bar_refresh_rate=10, max_epochs = 25, gpus=1)

trainer.fit(model, train_loader, val_loader)
trainer.test(model,test_loader)
pred = trainer.predict(model, test_loader) ## Would have a 4th dataloader
