import numpy as np
import SimpleITK as sitk
import cv2
from PIL import Image
import os

def get_pancreas_z_bounds(mask_array):
    """获取包含胰腺的z轴范围"""
    pancreas_region = (mask_array == 2)  # 胰腺的label是2
    z_indices = np.where(np.any(pancreas_region, axis=(1, 2)))[0]
    if len(z_indices) == 0:
        return None
    
    # 添加上下文窗口
    margin = 5
    z_min = max(0, z_indices.min() - margin)
    z_max = min(mask_array.shape[0], z_indices.max() + margin + 1)
    return z_min, z_max

def nii_to_gif(nii_path, output_path, axis=0, duration=50, show_frame=True, is_ct_image=False, mask_path=None):
    """
    将 .nii.gz 文件转换为 GIF 动画
    参数:
        nii_path: nii.gz 文件路径
        output_path: 输出路径
        axis: 切片方向
        duration: 帧持续时间
        show_frame: 是否显示帧数
        is_ct_image: 是否是CT图像
        mask_path: 如果是CT图像，需要提供对应的mask路径
    """
    # 读取文件
    img = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(img)

    if is_ct_image and mask_path:
        # 读取对应的mask文件
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # 获取胰腺的z轴范围
        z_bounds = get_pancreas_z_bounds(mask_array)
        if z_bounds:
            z_min, z_max = z_bounds
            array = array[z_min:z_max]
            mask_array = mask_array[z_min:z_max]
            print(f"Cropped z-axis from {z_min} to {z_max}")

    # 获取切片
    if axis == 0:
        slices = [array[i,:,:] for i in range(array.shape[0])]
        mask_slices = [mask_array[i,:,:] for i in range(mask_array.shape[0])] if is_ct_image else None
    elif axis == 1:
        slices = [array[:,i,:] for i in range(array.shape[1])]
        mask_slices = [mask_array[:,i,:] for i in range(mask_array.shape[1])] if is_ct_image else None
    else:  # axis == 2
        slices = [array[:,:,i] for i in range(array.shape[2])]
        mask_slices = [mask_array[:,:,i] for i in range(mask_array.shape[2])] if is_ct_image else None

    # 创建彩色切片
    rgb_slices = []
    total_frames = len(slices)
    
    for i, slice_array in enumerate(slices):
        if is_ct_image:
            # CT值窗口化处理
            window_center = 40
            window_width = 400
            ct_min = window_center - window_width/2
            ct_max = window_center + window_width/2
            slice_array = np.clip(slice_array, ct_min, ct_max)
            normalized = ((slice_array - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
            
            # 转换为RGB
            rgb_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            
            # 添加彩色标注
            mask_slice = mask_slices[i]
            # 半透明叠加
            overlay = rgb_img.copy()
            overlay[mask_slice == 2] = [0, 255, 0]  # 胰腺区域显示为绿色
            overlay[mask_slice == 1] = [255, 0, 0]  # 肿瘤区域显示为红色
            # 降低透明度到0.15
            rgb_img = cv2.addWeighted(overlay, 0.15, rgb_img, 0.85, 0)
        else:
            # 处理mask文件
            rgb_img = np.zeros((*slice_array.shape, 3), dtype=np.uint8)
            rgb_img[slice_array == 2] = [0, 255, 0]
            rgb_img[slice_array == 1] = [255, 0, 0]
        
        # 添加帧数
        if show_frame:
            cv2.putText(rgb_img,
                       f'{i+1:>03d}/{total_frames:>03d}',
                       (50, 50),
                       cv2.FONT_HERSHEY_COMPLEX,
                       1,
                       (255, 255, 255),
                       2)
        
        rgb_slices.append(rgb_img)

    # 转换为PIL图像
    images = [Image.fromarray(img) for img in rgb_slices]

    # 保存GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def main():
    # 输入输出路径
    ct_path = "/opt/data/private/wny/tumor_visualization/output/combined_img/final_combined_img.nii.gz"
    mask_path = "/opt/data/private/wny/tumor_visualization/output/combined_mask/final_combined_mask.nii.gz"
    output_dir = "/opt/data/private/wny/tumor_visualization/output/gifs"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成三个方向的GIF（带有彩色标注的CT图像）
    directions = ['sagittal', 'coronal', 'axial']
    for i, direction in enumerate(directions):
        # CT图像的GIF
        output_path = os.path.join(output_dir, f'final_combined_img_{direction}.gif')
        nii_to_gif(ct_path, output_path, axis=i, show_frame=True, is_ct_image=True, mask_path=mask_path)
        print(f"Generated CT GIF for {direction} view: {output_path}")

        # Mask的GIF
        output_path = os.path.join(output_dir, f'final_combined_mask_{direction}.gif')
        nii_to_gif(mask_path, output_path, axis=i, show_frame=True)
        print(f"Generated mask GIF for {direction} view: {output_path}")

if __name__ == "__main__":
    main()
