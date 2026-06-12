#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SWI-QSM 图像对验证脚本
用于评估 GAN 模型在 SWI 到 QSM 转换任务上的性能

使用方法:
    单例: python validate_swi_qsm.py --swi_path <SWI> --qsm_path <QSM> [--output_dir DIR]
    批量: python validate_swi_qsm.py --run_dataset ukb:<UKB根目录> --run_dataset bids:<BIDS根目录> \\
          [--aggregate_csv all.csv] [--batch_save_nifti]
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import SimpleITK as sitk
import nibabel as nib
from scipy import stats
import json
import csv
import pandas as pd
from jax import numpy as jnp
import copy
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from model.unet import UNetSmall
from model.discriminator import Discriminator_ccgan
from utils import metrics_2d, metrics_3d_jnp,metrics_3d
from utils.postprocessing import zscore, scaling_to_minus1_1, minmax_normalization, robust_minmax_from_gt

class SpacingNotAlign(Exception):
    print("Given two images have different matrix size, please check")
    pass


def _iter_ukb_subject_pairs(root):
    """
    UKB 布局（与命令行说明一致）:
      {root}/{subject_id}/SWI.nii.gz
      {root}/{subject_id}/QSM/QSM.nii.gz  （若不存在则回退到同目录 QSM.nii.gz）
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        subdir = os.path.join(root, name)
        if not os.path.isdir(subdir):
            continue
        swi = os.path.join(subdir, "SWI.nii.gz")
        qsm_nested = os.path.join(subdir, "QSM", "QSM.nii.gz")
        qsm_flat = os.path.join(subdir, "QSM.nii.gz")
        if not os.path.isfile(swi):
            continue
        if os.path.isfile(qsm_nested):
            qsm = qsm_nested
        elif os.path.isfile(qsm_flat):
            qsm = qsm_flat
        else:
            continue
        yield name, swi, qsm


def _iter_bids_subject_pairs(root):
    """
    BIDS 风格:
      {root}/sub-xxx/SWI/sub-xxx_SWI.nii.gz
      {root}/sub-xxx/QSM/result/QSM.nii.gz
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")
    for name in sorted(os.listdir(root)):
        if not name.startswith("sub-") or name.startswith("._"):
            continue
        subdir = os.path.join(root, name)
        if not os.path.isdir(subdir):
            continue
        swi = os.path.join(subdir, "SWI", f"{name}_SWI.nii.gz")
        qsm = os.path.join(subdir, "QSM", "result", "QSM.nii.gz")
        if os.path.isfile(swi) and os.path.isfile(qsm):
            yield name, swi, qsm


def _flatten_metrics_to_row(metrics, subject_id, cohort=None):
    flattened = {}
    if cohort is not None:
        flattened["cohort"] = cohort
    flattened["subject_id"] = subject_id
    for k, v in metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                col_name = f"{k}_{k2}" if k2 else k
                if isinstance(v2, (np.integer, np.floating, np.ndarray)):
                    if isinstance(v2, np.ndarray):
                        flattened[col_name] = float(v2.item() if v2.size == 1 else v2.mean())
                    else:
                        flattened[col_name] = float(v2)
                elif isinstance(v2, (int, float)):
                    flattened[col_name] = float(v2)
                else:
                    flattened[col_name] = v2
        elif isinstance(v, (np.integer, np.floating, np.ndarray)):
            if isinstance(v, np.ndarray):
                flattened[k] = float(v.item() if v.size == 1 else v.mean())
            else:
                flattened[k] = float(v)
        elif isinstance(v, (int, float)):
            flattened[k] = float(v)
        else:
            flattened[k] = v
    return flattened


def _append_metrics_csv(csv_path, flattened_metrics):
    file_exists = os.path.exists(csv_path)
    df = pd.DataFrame([flattened_metrics])
    if file_exists:
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False, encoding="utf-8")


class SWIQSMValidator:
    """SWI-QSM 验证器"""
    
    def __init__(self, gen_model_path=None, disc_model_path=None, device=None):
        """
        初始化验证器
        
        Args:
            gen_model_path: 生成器模型路径，默认使用项目中的模型
            disc_model_path: 判别器模型路径，默认使用项目中的模型
            device: 计算设备，默认自动选择
        """
        # 设置设备
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 默认模型路径
        if gen_model_path is None:
            gen_model_path = os.path.join(project_root, 'res/save/202403/gen_new_19.pt')
        if disc_model_path is None:
            disc_model_path = os.path.join(project_root, 'res/save/202403/disc_new_19.pt')
        
        self.input_direction = np.array([[-1, 0, 0],[0,-1,0],[0,0,1]])
        # 加载模型
        self.gen = self._load_generator(gen_model_path)
        self.disc = self._load_discriminator(disc_model_path)


    def _load_generator(self, model_path):
        """加载生成器模型"""
        print(f"加载生成器模型: {model_path}")
        gen = UNetSmall(num_channels=1, num_classes=1, normalize=False)
        gen = self._load_state_dict(gen, model_path)
        gen = gen.to(self.device).eval()
        return gen
    
    def _load_discriminator(self, model_path):
        """加载判别器模型"""
        print(f"加载判别器模型: {model_path}")
        disc = Discriminator_ccgan([1,256,256])
        disc = self._load_state_dict(disc, model_path)
        disc = disc.to(self.device).eval()
        return disc
    
    def _load_state_dict(self, model, state_dict_path):
        """加载模型权重，处理 DataParallel 的情况"""
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'), weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model
    
    def load_nifti_image(self, path):
        """加载 NII 格式图像"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        # 使用 SimpleITK 加载
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)  # 形状: (D, H, W)
        array = np.asarray(array,dtype=np.float32)
        return array, image
    
    def apply_mask(self, img, mask):
        '''
        Apply mask to the image
        Args:
            img: SimpleITK image to apply mask
            mask: SimpleITK image mask to apply
        Returns:
            out: SimpleITK image with mask applied
        Raises:
            SpacingNotAlign: if the image and mask have different matrix size
        '''

        mask_on= sitk.Resample(
            mask,                   # moving image
            img,                    # reference image (defines size/origin/spacing/direction)
            sitk.Transform(),       # identity
            sitk.sitkNearestNeighbor,  # IMPORTANT for masks/labels
            0,                      # default background
            sitk.sitkUInt8
        )
        # Ensure binary (optional, but recommended)
        mask_on = mask_on > 0
        # Apply
        out = sitk.Mask(img, mask_on)   # outside mask -> 0 by default
        return out 

    def interpolate_and_flip(self, image, newSize, flipAxes = [True,False,True]):
        """
        Interpolate and reverse the image to the new size
        Args:
            image: SimpleITK image to interpolate and reverse
            newSize: new size of the image (list, tuple, or numpy array)
            if_filp: Whether done file
            flipAxes: axes to flip
        Returns:
            SimpleITK image: interpolated and reversed image
        """
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkLinear) 	##此处为线性插值，其他插值方式可以去官网查询
        resampleFilter.SetOutputDirection(image.GetDirection())
        resampleFilter.SetOutputOrigin(image.GetOrigin())
        
        # 确保 newSize 是整数元组
        if isinstance(newSize, np.ndarray):
            newSize = tuple(int(x) for x in newSize)
        elif isinstance(newSize, list):
            newSize = tuple(int(x) for x in newSize)
        else:
            newSize = tuple(int(x) for x in newSize)
        
        oriSize = np.array(image.GetSize(), dtype=np.float64)
        oriSpacing = np.array(image.GetSpacing(), dtype=np.float64)
        newSize_array = np.array(newSize, dtype=np.float64)
        
        # 计算新的 spacing
        newSpacing = (oriSize / newSize_array) * oriSpacing
        newSpacing = tuple(float(x) for x in newSpacing)
        
        resampleFilter.SetSize(newSize)
        resampleFilter.SetOutputSpacing(newSpacing)
        newImage = resampleFilter.Execute(image)
        #flip along z
        fliped = sitk.Flip(newImage, flipAxes = flipAxes, flipAboutOrigin=True)
        return fliped
    
    def preprocess_image(self, image_array, target_size=256):
        """
        Preprocess the image to the target size
        Args:
            image_array: numpy array of the image
            target_size: target size of the image
        Returns:
            torch tensor: preprocessed image, shape (1,D,target_size,target_size)
        """
        # SimpleITK GetArrayFromImage 输出已是 (D,H,W)
        x = np.asarray(image_array, dtype=np.float32)
        x = torch.from_numpy(x).contiguous()  # (D,H,W)

        D, H, W = x.shape
        S = max(H, W)

        # pad to SxS (pads last two dims: W then H)
        pad_h = S - H
        pad_w = S - W
        left  = pad_w // 2
        right = pad_w - left
        top   = pad_h // 2
        bottom= pad_h - top
        x = F.pad(x, (left, right, top, bottom), mode="constant", value=0)  # (D,S,S)

        # resize to 256x256 if needed (this changes spacing)
        if S != target_size:
            x = transforms.Resize(size=target_size, antialias=True)(x)
        x = transforms.CenterCrop(target_size)(x)
        return x.unsqueeze(0)  # (1,D,256,256)
    
    def generate_qsm(self, swi_tensor):
        """
        使用生成器从 SWI 生成 QSM
        
        Args:
            swi_tensor: SWI 图像张量，形状为 (D, 256, 256)
        
        Returns:
            torch.tensor: 生成的 QSM,形状为 (D, 256, 256)
        """
        # 确保数据类型为 float32
        swi_tensor = swi_tensor.float()
        swi_tensor = swi_tensor.unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            qsm_fake = self.gen(swi_tensor)  # (D, 1, 256, 256)
            qsm_fake_cpu = qsm_fake.cpu().squeeze().detach() # (D, 256, 256)
        
        return qsm_fake_cpu

    def cal_metrics(self, real_img, fake_img, dim, mask=None):
        if real_img.shape != fake_img.shape:
            raise SpacingNotAlign
        if dim == 3:
            real_np = np.expand_dims(np.asarray(real_img), axis=0)
            fake_np = np.expand_dims(np.asarray(fake_img), axis=0)
            mask_np = None
            if mask is not None:
                mask_np = np.asarray(mask)
                # 3D mask -> add batch axis to match (1, D, H, W)
                if mask_np.ndim == real_np.ndim - 1:
                    mask_np = np.expand_dims(mask_np, axis=0)
            print(real_np.shape, fake_np.shape)
            metrics_3d_result = metrics_3d.cal_metrics(real_np, fake_np, mask_np)
            metrics_3d_jnp_result = metrics_3d_jnp.cal_metrics(jnp.array(real_np), jnp.array(fake_np))
            return {**metrics_3d_result, **metrics_3d_jnp_result}
        elif dim == 2:
            real_np = np.asarray(real_img)
            fake_np = np.asarray(fake_img)
            mask_np = None if mask is None else np.asarray(mask, dtype=bool)
            return metrics_2d.cal_metrics(real_np, fake_np, mask_np)
        else:
            raise ValueError(f"Invalid score measure dimension: {dim}")
    
    def generate_only(self, swi_path, output_dir):
        """
        使用生成器从 SWI 生成 QSM
        """
        print(f"\n加载 SWI 图像: {swi_path}")
        _, swi_sitk = self.load_nifti_image(swi_path)
        print(f"SWI 图像形状: {swi_sitk.GetSize()}")
        swi_tensor = self.preprocess_image(sitk.GetArrayFromImage(swi_sitk))
        print(f"swi_tensor shape: {swi_tensor.shape}")
        qsm_fake = self.generate_qsm(swi_tensor)
        print(f"qsm_fake shape: {qsm_fake.shape}, swi_sitk shape: {swi_sitk.GetSize()}")
        qsm_fake_sitk = sitk.GetImageFromArray(qsm_fake)
        qsm_fake_sitk.CopyInformation(swi_sitk)
        sitk.WriteImage(qsm_fake_sitk, os.path.join(output_dir, 'qsm_fake.nii.gz'))
        sitk.WriteImage(swi_sitk, os.path.join(output_dir, 'swi_true.nii.gz'))
        print(f"\n生成的 QSM 已保存到: {output_dir}")
        

    def save_gen_result(self, qsm_fake, ref, output_dir, filename="qsm_fake_align.nii.gz"):
        """
        Inverse of (pad-to-square -> resize-to-256):
        qsm_fake: torch Tensor (1, D, 256, 256) or (D, 256, 256)
        ref: SimpleITK image with original geometry/size (W,H,D)
        Saves: (H,W,D) content restored in size, written with ref metadata.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ---- get original size from ref (SimpleITK uses (W,H,D)) ----
        ref_W, ref_H, ref_D = ref.GetSize()

        # ---- normalize tensor shape to (D,H,W) ----
        x = qsm_fake
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        else:
            raise TypeError("qsm_fake must be a torch.Tensor")

        if x.ndim == 4:
            # (1,D,H,W) expected
            if x.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {x.shape}")
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected (D,H,W) or (1,D,H,W), got {tuple(x.shape)}")

        D, H, W = x.shape

        # Optional sanity: match depth if needed
        if D != ref_D:
            raise ValueError(f"Depth mismatch: tensor D={D}, ref D={ref_D}")

        # ---- invert preprocessing: resize 256->S, then center-crop to (ref_H,ref_W) ----
        S = max(ref_H, ref_W)

        # resize to SxS if needed (spatial dims only)
        if (H, W) != (S, S):
            x = transforms.Resize(size=S, antialias=True)(x)

        # center crop to original (H,W)
        x = transforms.CenterCrop((ref_H, ref_W))(x)  # result (D, ref_H, ref_W)

        # ---- write sitk image (GetImageFromArray expects (z,y,x) = (D,H,W)) ----
        arr = x.numpy().astype("float32")
        out_img = sitk.GetImageFromArray(arr)   # size becomes (W,H,D)
        out_img.CopyInformation(ref)            # copies origin/spacing/direction
        sitk.WriteImage(out_img, os.path.join(output_dir, filename))


    def validate(self, swi_path, qsm_path, method ,if_mask=False, mask_path=None, output_dir=None, flipAxes = [False,False,False]):
        """
        验证 SWI-QSM 图像对
        
        Args:
            swi_path: SWI 图像路径
            qsm_path: 真实 QSM 图像路径（用于计算评价指标）
            method: 验证方法
            if_mask: 是否应用掩码
            mask_path: 掩码图像路径，如果为 None 则不应用掩码
            output_dir: 输出目录，如果指定则保存生成的 QSM
            flipAxes: 翻转轴
        
        Returns:
            dict: 包含所有评价指标的字典（包括处理时间）
        """
        # 开始计时
        start_time = time.time()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"\n加载 SWI 图像: {swi_path}")
        _, swi_sitk = self.load_nifti_image(swi_path)
        print(f"SWI 图像形状: {swi_sitk.GetSize()}")
        
        if qsm_path is None:
            print("Only validate SWI image")
            self.generate_only(swi_path, output_dir)
            return
        
        print(f"\n加载 QSM 图像: {qsm_path}")
        _, qsm_sitk = self.load_nifti_image(qsm_path)
        print(f"QSM 图像形状: {qsm_sitk.GetSize()}")

        if if_mask and mask_path is not None:
            _, mask_sitk = self.load_nifti_image(mask_path)
            
        # 预处理
        print("\n预处理图像...")
        # do interpolation and reverse
        swi_size = swi_sitk.GetSize()
        qsm_size = qsm_sitk.GetSize()
        target_size = swi_size
        
        if qsm_size != target_size:
            print(f"QSM 图像形状不匹配，进行插值和反转: {qsm_size} -> {target_size}")
            qsm_sitk = self.interpolate_and_flip(qsm_sitk, target_size, flipAxes = flipAxes)
        if if_mask and mask_sitk is not None and mask_sitk.GetSize() != target_size:
            print(f"掩码图像形状不匹配，进行插值和反转: {mask_sitk.GetSize()} -> {target_size}")
            mask_sitk = self.interpolate_and_flip(mask_sitk, target_size, flipAxes = flipAxes)
            print(mask_sitk.GetSize()==swi_sitk.GetSize())
            swi_sitk = self.apply_mask(swi_sitk, mask_sitk)
            qsm_sitk = self.apply_mask(qsm_sitk, mask_sitk)
            
        qsm_tensor = sitk.GetArrayFromImage(qsm_sitk)

        qsm_sitk_new = qsm_sitk
        if qsm_tensor.min() > -2 or qsm_tensor.max() < 2:
            print('QSM in ppm, switch to ppb')
            qsm_tensor = (qsm_tensor.astype(np.float32) * 1000)
            qsm_sitk_new = sitk.GetImageFromArray(qsm_tensor)
            qsm_sitk_new.CopyInformation(qsm_sitk)
            if output_dir:
                sitk.WriteImage(qsm_sitk_new, os.path.join(output_dir, 'qsm_true_ppm.nii.gz'))
            qsm_tensor = qsm_tensor.astype(np.float32)

        swi_tensor = self.preprocess_image(sitk.GetArrayFromImage(swi_sitk))
        qsm_tensor = self.preprocess_image(sitk.GetArrayFromImage(qsm_sitk_new))
        mask_tensor = self.preprocess_image(sitk.GetArrayFromImage(mask_sitk)) if if_mask and mask_sitk is not None else None
        
        print(f"预处理后 SWI 形状: {swi_tensor.shape}") # (1,D,W,H)
        print(f"预处理后 QSM 形状: {qsm_tensor.shape}")
        
        # 生成 QSM
        print("\n使用生成器生成 QSM...")
        # preprocess_image 返回 (1,D,H,W)，生成器按切片维度推理需要 (D,H,W)
        qsm_fake = self.generate_qsm(swi_tensor.squeeze(0))
        qsm_real = qsm_tensor.squeeze(0) # (D,H,W) on cpu
        if mask_tensor is not None:
            mask_tensor = mask_tensor.squeeze(0)
        
        # 转换为 numpy 数组用于计算指标
        metrics = {}
        if method == 'origin':
            print('using origin method')
            qsm_real_ts = qsm_real
            qsm_fake_ts = qsm_fake
        elif method == 'zscore':
            print('using zscore method')
            qsm_real_ts = zscore(qsm_real)
            qsm_fake_ts = zscore(qsm_fake)
        elif method == 'robust_minmax':
            print('using robust minmax method')
            qsm_real_ts, qsm_fake_ts = robust_minmax_from_gt(qsm_real, qsm_fake)
        else:
            raise ValueError(f"Invalid method: {method}")

        print('qsm_real minimun',qsm_real_ts.min(),'qsm_real maximun',qsm_real_ts.max())
        print('qsm_fake minimun',qsm_fake_ts.min(),'qsm_fake maximun',qsm_fake_ts.max())

        print("\n计算评价指标...")
        # 计算所有指标
        metrics_result_3d = self.cal_metrics(qsm_real_ts, qsm_fake_ts, dim=3, mask=mask_tensor)
        metrics_result_2d = self.cal_metrics(qsm_real_ts, qsm_fake_ts, dim=2, mask=mask_tensor)
        metrics= {**metrics_result_3d, **metrics_result_2d}

        metrics['qsm_real_minimun'] = float(qsm_real_ts.min())
        metrics['qsm_real_maximun'] = float(qsm_real_ts.max())
        metrics['qsm_fake_minimun'] = float(qsm_fake_ts.min())
        metrics['qsm_fake_maximun'] = float(qsm_fake_ts.max())

        # 保存生成的 QSM
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            self.save_gen_result(qsm_fake_ts, swi_sitk, output_dir)
            sitk.WriteImage(qsm_sitk, os.path.join(output_dir, 'qsm_true.nii.gz'))
            sitk.WriteImage(swi_sitk, os.path.join(output_dir, 'swi_true.nii.gz'))
        
        # 计算总处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        metrics['processing_time_seconds'] = float(processing_time)
        print(f"\n处理时间: {processing_time:.2f} 秒")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(
        description='验证 SWI-QSM 图像对的 GAN 模型性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    示例:
    python validate_swi_qsm.py --swi_path /path/to/swi.nii.gz --qsm_path /path/to/qsm.nii.gz
    python validate_swi_qsm.py --swi_path /path/to/swi.nii.gz --qsm_path /path/to/qsm.nii.gz --output_dir ./results
    批量（UKB train/val + BIDS 测试集）:
    python validate_swi_qsm.py --run_dataset ukb:/path/to/train --run_dataset ukb:/path/to/validate \\
      --run_dataset bids:/path/to/csvd-bids --run_dataset bids:/path/to/shunyi_followup_bids \\
      --aggregate_csv ./metrics_all.csv
            """
    )
    
    parser.add_argument('--swi_path', type=str, default=None,
                       help='SWI 图像文件路径 (NII 格式)；与 --run_dataset 二选一')
    parser.add_argument('--generate_only', action='store_true', default=False,
                       help='仅生成 QSM 图像')
    parser.add_argument('--qsm_path', type=str, default=None,
                       help='真实 QSM 图像文件路径 (NII 格式，用于计算评价指标)')
    parser.add_argument('--if_mask', action='store_true', default=False,
                       help='是否应用掩码')
    parser.add_argument('--mask_path', type=str, default=None,
                       help='掩码图像文件路径 (NII 格式，用于去脑壳)')
    parser.add_argument('--method', type=str, default='robust_minmax',
                       choices=['origin', 'zscore', 'robust_minmax'],
                       help='验证方法 (默认: robust_minmax)')
    parser.add_argument('--gen_model_path', type=str, default=None,
                       help='生成器模型路径 (默认: res/save/202403/gen_new_19.pt)')
    parser.add_argument('--disc_model_path', type=str, default=None,
                       help='判别器模型路径 (默认: res/save/202403/disc_new_19.pt)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，用于保存生成的 QSM 图像')
    parser.add_argument('--save_metrics', type=str, default=None,
                       help='保存评价指标到 CSV 文件')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='计算设备 (默认: 自动选择)')
    parser.add_argument('--subject_id', type=str, default=None,
                       help='被试ID，用于保存到CSV文件中')
    parser.add_argument(
        '--run_dataset',
        action='append',
        default=None,
        metavar='LAYOUT:ROOT',
        help='批量评估（可重复）。LAYOUT=ukb|bids；ROOT 为数据根目录。'
             ' ukb: {id}/SWI.nii.gz 与 {id}/QSM/QSM.nii.gz（若无则用 {id}/QSM.nii.gz）。'
             ' bids: sub-*/SWI/sub-*_SWI.nii.gz 与 sub-*/QSM/result/QSM.nii.gz',
    )
    parser.add_argument(
        '--output_base',
        type=str,
        default=None,
        help='批量模式：输出根目录（每例写入 {output_base}/{队列名}/{subject_id}/），默认 ./batch_validate_out',
    )
    parser.add_argument(
        '--aggregate_csv',
        type=str,
        default=None,
        help='批量模式：将所有 subject 的指标汇总追加到该 CSV（推荐与 --run_dataset 同用）',
    )
    parser.add_argument(
        '--batch_save_nifti',
        action='store_true',
        default=False,
        help='批量模式：为每例保存生成的 NIfTI；默认仅算指标以节省时间与磁盘',
    )
    
    args = parser.parse_args()

    batch_mode = bool(args.run_dataset)
    if not batch_mode and not args.swi_path:
        parser.error('请提供 --swi_path，或使用 --run_dataset 进行批量评估')
    if batch_mode and args.swi_path:
        parser.error('批量模式请勿同时指定 --swi_path')
    
    # 创建验证器
    validator = SWIQSMValidator(
        gen_model_path=args.gen_model_path,
        disc_model_path=args.disc_model_path,
        device=args.device,
    )

    if args.generate_only:
        if not args.swi_path:
            parser.error("--generate_only 需要 --swi_path")
        validator.generate_only(args.swi_path, args.output_dir)
        sys.exit(0)

    if batch_mode:
        output_base = args.output_base or os.path.join(project_root, "batch_validate_out")
        csv_out = args.aggregate_csv or args.save_metrics

        try:
            for spec in args.run_dataset:
                if ":" not in spec:
                    parser.error(f"无效的 --run_dataset: {spec}，应为 LAYOUT:ROOT")
                layout, root = spec.split(":", 1)
                layout = layout.strip().lower()
                root = root.strip()
                if layout == "ukb":
                    pairs = list(_iter_ukb_subject_pairs(root))
                elif layout == "bids":
                    pairs = list(_iter_bids_subject_pairs(root))
                else:
                    parser.error(f"未知 LAYOUT: {layout}，应为 ukb 或 bids")

                cohort = os.path.basename(os.path.normpath(root))
                print(f"\n=== 队列 {cohort} ({layout}) 共 {len(pairs)} 例 ===")

                for subject_id, swi_path, qsm_path in pairs:
                    per_out = None
                    if args.batch_save_nifti:
                        per_out = os.path.join(output_base, cohort, subject_id)
                        os.makedirs(per_out, exist_ok=True)
                    try:
                        metrics = validator.validate(
                            swi_path=swi_path,
                            qsm_path=qsm_path,
                            method=args.method,
                            if_mask=args.if_mask,
                            mask_path=args.mask_path,
                            output_dir=per_out,
                            flipAxes=[False, False, False],
                        )
                    except Exception as e:
                        print(f"[跳过] {cohort}/{subject_id}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        continue
                    row = _flatten_metrics_to_row(metrics, subject_id, cohort=cohort)
                    if csv_out:
                        _append_metrics_csv(csv_out, row)
                    print(f"  OK {subject_id}")

            print("\n批量验证完成!")
            if csv_out:
                print(f"汇总 CSV: {csv_out}")
        except Exception as e:
            print(f"\n错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    # 执行验证（单例）
    try:
        metrics = validator.validate(
            swi_path=args.swi_path,
            qsm_path=args.qsm_path,
            method=args.method,
            if_mask=args.if_mask,
            mask_path=args.mask_path,
            output_dir=args.output_dir,
            flipAxes=[False, False, False],
        )

        if args.save_metrics:
            if args.subject_id:
                subject_id = args.subject_id
            else:
                subject_id = os.path.basename(os.path.dirname(args.swi_path))
            flattened_metrics = _flatten_metrics_to_row(metrics, subject_id, cohort=None)
            _append_metrics_csv(args.save_metrics, flattened_metrics)
            print(f"\n评价指标已保存到: {args.save_metrics}")

        print("\n验证完成!")

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

