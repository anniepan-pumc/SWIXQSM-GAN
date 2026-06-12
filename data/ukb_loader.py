import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from data.nifti_tools import load_nifti, show_nifti
# from nifti_tools import load_nifti, show_nifti
from PIL import Image

transform = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((256,256),interpolation=Image.Resampling.NEAREST),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])


class UKBDataset(Dataset):
    def __init__(self, data_path, record_path, SWI_path, QSM_path, transform=transform, rand_slice=False):
        ''''
        Create dataset for SWI2QSM 
        :param data_path: Data folder
        :param record_path: Data record
        :param SWI_path: Path where save SWI data / features
        :param QSM_path: Path where save QSM data / target
        :param if_trans: transformation for image data; default True
        :param Padding: Whether padding on each side of imput data to balance height and width; default True
        :param rand_slice: random pick data over slices of data if True else over img
        '''
        self.data_path = data_path
        self.record_path = record_path
        self.rand_slice = rand_slice
        if SWI_path == '':
            self.SWI_path = data_path
        else:
            self.SWI_path = os.path.join(data_path, SWI_path)
        if QSM_path == '':
            self.QSM_path = data_path
        else:
            self.QSM_path = os.path.join(data_path, QSM_path)
        # self.mask_path = os.path.join(data_path, mask_path)
        self.transform = transform

        df = pd.read_csv(self.record_path, dtype={'SWI': str, 'QSM': str, "InBoth": bool})
        df = df.loc[df['InBoth'] == 1][['SWI']].rename(columns={'SWI': "ID"})
        # df = pd.read_csv(self.record_path, dtype={'ID':str, 'Label':bool})
        self.data_record = self.data_slice(df) if self.rand_slice else df

    def data_slice(self, df):
        '''
        mix up data into slice
        :param df:
        :return: new df
        '''      
        return df.merge(pd.DataFrame({"slice": range(48)}), how='cross')

    def __len__(self):
        return self.data_record.shape[0]

    def __getitem__(self, idx):
        row = self.data_record.iloc[idx]
        swi = load_nifti(os.path.join(self.SWI_path, row['ID'], "SWI.nii.gz")).permute(2, 1, 0).to(torch.float32)
        qsm = load_nifti(os.path.join(self.QSM_path, row['ID'], "QSM.nii.gz")).permute(2, 1, 0).to(torch.float32)
        # mask = load_nifti(os.path.join(self.mask_path, row['ID'], "QSM_mask.nii.gz")).permute(2, 1, 0).to(torch.float32)

        if self.rand_slice:
            swi = swi[row['slice'], :, :].unsqueeze(0)
            qsm = qsm[row['slice'], :, :].unsqueeze(0)

        swi_t = torch.empty([swi.shape[0],256,256])
        qsm_t = torch.empty([qsm.shape[0],256,256])
        if self.transform is not None:
            for i in range(swi.shape[0]):
                swi_t[i] = self.transform(swi[i,:,:])
                qsm_t[i] = self.transform(qsm[i,:,:])
        # print('dataloader: ',swi.shape,qsm.shape) # torch.Size([48 288,256]) torch.Size([48,288,256])
        del swi,qsm
        return idx, swi_t, qsm_t

    def show_nifti(self, idx, col_num=6, fig_size=15, save_path='./save'):
        ID, n_slice = self.data_record.iloc[idx]
        show_nifti(load_nifti(os.path.join(self.SWI_path, ID, "SWI.nii.gz")).permute(2, 1, 0),
                   col_num, fig_size, os.path.join(save_path, ID, "SWI"))
        show_nifti(load_nifti(os.path.join(self.QSM_path, ID, "QSM.nii.gz")).permute(2, 1, 0),
                   col_num, fig_size, os.path.join(save_path, ID, "QSM"))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps" if torch.backends.mps.is_available() else "cpu" # for Mac Apple silicon
    data_record = pd.read_csv('data/ukb_data_summary.csv', dtype={'SWI': str, 'QSM': str, 'InBoth': int})
    data_path = '/mnt/data/ukb_swi/nifti'
    record_path = 'data/ukb_data_summary.csv'
    SWI_path = 'SWI'
    QSM_path = 'QSM'
    # Create dataset and data loader
    dataset = UKBDataset(data_path=data_path, record_path=record_path, SWI_path=SWI_path, QSM_path=QSM_path,
                            transform=transform, rand_slice=True)
    batch_size = 100
    shuffle = True
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i,data in enumerate(data_loader):
        if i == 15:
            idx, swi, qsm = data
            print(idx, swi.shape, qsm.shape)
            print(swi.min(),swi.max())
            print(qsm.min(),qsm.max())
            break
