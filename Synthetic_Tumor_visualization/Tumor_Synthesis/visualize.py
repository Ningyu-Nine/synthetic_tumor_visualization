from collections import defaultdict
import numpy as np
from random import choice
from cellular.cellular import update_cellular
import torch
import argparse
import SimpleITK as sitk
from skimage.measure import label, regionprops
import cv2
import math
import os
from utils_visual import generate_tumor, get_predefined_texture

Organ_List = {'pancreas': [1,4]}
Organ_HU = {'pancreas': [100, 160]}

def get_user_steps():
    """Get user input for number of steps"""
    while True:
        try:
            steps = int(input("\nEnter total steps for tumor growth (recommended 5-70): "))
            if steps > 0:
                return steps
            print("Steps must be greater than 0")
        except ValueError:
            print("Please enter a valid integer")

def main():
    steps = get_user_steps()
    
    organ_name = 'pancreas'  # organ name
    start_point = (0, 0, 0)  # start point
    save_base_dir = '/opt/data/private/wny/tumor_visualization/output'
    
    # 为图像和标签创建单独的目录
    save_img_dir = os.path.join(save_base_dir, 'growth_process/images')
    save_label_dir = os.path.join(save_base_dir, 'growth_process/labels')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)
    
    # Adjust save interval based on total steps
    save_interval = max(1, steps // 10)  # Default to saving about 10 intermediate results
    print(f"Will save results every {save_interval} steps")

    organ_hu_lowerbound = Organ_HU[organ_name][0]  # organ hu lowerbound
    outrange_standard_val = Organ_HU[organ_name][1]  # outrange standard value
    organ_standard_val = 0  # organ standard value
    threshold = 10  # threshold
    kernel_size = (3, 3, 3)  # Receptive Field
    
    textures = []
    sigma_as = [3, 6, 9, 12, 15]
    sigma_bs = [4, 7]
    predefined_texture_shape = (500, 500, 500)
    for sigma_a in sigma_as:
        for sigma_b in sigma_bs:
            texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
            textures.append(texture)
    print("All predefined textures have been generated.")
    
    # 选择一个纹理
    texture = textures[0]
    
    # 读取原始图像时保存元数据信息
    original_img = sitk.ReadImage('/opt/data/private/wny/tumor_visualization/image/100000_00001_0000.nii.gz')
    original_spacing = original_img.GetSpacing()
    original_origin = original_img.GetOrigin()
    original_direction = original_img.GetDirection()
    
    img = sitk.GetArrayFromImage(original_img)
    label = sitk.GetArrayFromImage(sitk.ReadImage('/opt/data/private/wny/tumor_visualization/label/100000_00001.nii.gz'))
    
    # 设置GPU参数
    args = argparse.Namespace(gpu=0)  # 直接创建一个包含gpu属性的对象
    
    # 修改generate_tumor调用，添加return_intermediate=True
    img, label, intermediate_imgs, intermediate_masks = generate_tumor(
        img,
        label, 
        texture, 
        steps, 
        kernel_size, 
        organ_standard_val, 
        organ_hu_lowerbound, 
        outrange_standard_val, 
        threshold, 
        organ_name, 
        start_point,
        args,
        return_intermediate=True  # 返回中间结果
    )
    
    # 保存最终结果
    output_img = sitk.GetImageFromArray(img)
    output_img.SetSpacing(original_spacing)
    output_img.SetOrigin(original_origin)
    output_img.SetDirection(original_direction)
    
    output_label = sitk.GetImageFromArray(label)
    output_label.SetSpacing(original_spacing)
    output_label.SetOrigin(original_origin)
    output_label.SetDirection(original_direction)
    
    sitk.WriteImage(output_img, os.path.join(save_img_dir, 'final_tumor.nii.gz'))
    sitk.WriteImage(output_label, os.path.join(save_label_dir, 'final_label_tumor.nii.gz'))
    
    # 保存中间结果
    for i in range(0, len(intermediate_imgs), save_interval):
        step_num = i  # 步骤编号从1开始
        
        # 创建中间结果的SimpleITK图像
        intermediate_img = sitk.GetImageFromArray(intermediate_imgs[i])
        intermediate_img.SetSpacing(original_spacing)
        intermediate_img.SetOrigin(original_origin)
        intermediate_img.SetDirection(original_direction)
        
        intermediate_label = sitk.GetImageFromArray(intermediate_masks[i])
        intermediate_label.SetSpacing(original_spacing)
        intermediate_label.SetOrigin(original_origin)
        intermediate_label.SetDirection(original_direction)
        
        # 分别保存到各自目录
        sitk.WriteImage(intermediate_img, os.path.join(save_img_dir, f'tumor_step_{step_num:03d}.nii.gz'))
        sitk.WriteImage(intermediate_label, os.path.join(save_label_dir, f'tumor_step_{step_num:03d}.nii.gz'))
        
        print(f"保存了第 {step_num} 步的中间结果")
    
    print(f"所有图像已保存到: {save_img_dir}")
    print(f"所有标签已保存到: {save_label_dir}")
    
if __name__ == "__main__":
    main()
