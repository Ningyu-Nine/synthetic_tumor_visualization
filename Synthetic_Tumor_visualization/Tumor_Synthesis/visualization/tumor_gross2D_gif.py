import numpy as np
import SimpleITK as sitk
import cv2
from PIL import Image
import os
from skimage import measure

def find_largest_tumor_slice(ct_array, mask_array, axis=0):
    """找到肿瘤最大的切片
    axis: 0 for sagittal, 1 for coronal, 2 for axial
    """
    tumor_areas = []
    if axis == 0:
        for i in range(mask_array.shape[0]):
            tumor_area = np.sum(mask_array[i] == 1)
            tumor_areas.append(tumor_area)
    elif axis == 1:
        for i in range(mask_array.shape[1]):
            tumor_area = np.sum(mask_array[:, i, :] == 1)
            tumor_areas.append(tumor_area)
    else:  # axis == 2
        for i in range(mask_array.shape[2]):
            tumor_area = np.sum(mask_array[:, :, i] == 1)
            tumor_areas.append(tumor_area)
    
    max_slice_idx = np.argmax(tumor_areas)
    return max_slice_idx

def get_slice_from_volume(volume, mask, slice_idx, axis=0):
    """根据指定方向获取切片"""
    if axis == 0:
        return volume[slice_idx], mask[slice_idx]
    elif axis == 1:
        return volume[:, slice_idx, :], mask[:, slice_idx, :]
    else:  # axis == 2
        return volume[:, :, slice_idx], mask[:, :, slice_idx]

def get_mask_contours(mask_slice):
    """获取mask的轮廓"""
    # 使用 OpenCV 的轮廓检测替代 skimage
    mask_slice = mask_slice.astype(np.uint8)
    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_side_by_side_view(ct_slice, mask_slice, window_center=40, window_width=400):
    """创建原始CT和带标注CT的并排视图"""
    # CT值窗口化处理
    ct_min = window_center - window_width/2
    ct_max = window_center + window_width/2
    ct_slice = np.clip(ct_slice, ct_min, ct_max)
    normalized = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
    
    # 创建两个RGB图像
    original_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    annotated_rgb = original_rgb.copy()
    
    # 分别获取胰腺和肿瘤的轮廓
    pancreas_binary = (mask_slice == 2).astype(np.uint8)
    tumor_binary = (mask_slice == 1).astype(np.uint8)
    
    # 获取轮廓
    pancreas_contours, _ = cv2.findContours(pancreas_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tumor_contours, _ = cv2.findContours(tumor_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制轮廓
    cv2.drawContours(annotated_rgb, pancreas_contours, -1, (0, 255, 0), 1)  # 绿色轮廓表示胰腺
    cv2.drawContours(annotated_rgb, tumor_contours, -1, (255, 0, 0), 1)  # 红色轮廓表示肿瘤
    
    # 创建并排显示的图像
    combined_img = np.hstack([original_rgb, annotated_rgb])
    return combined_img

def create_tumor_growth_gif(ct_steps_dir, mask_steps_dir, output_path, axis=0, duration=500):
    """
    创建展示肿瘤生长过程的GIF
    参数:
        ct_steps_dir: 存放每个step的CT图像目录
        mask_steps_dir: 存放每个step的mask目录
        output_path: 输出的GIF路径
        axis: 0 for sagittal, 1 for coronal, 2 for axial
        duration: 每帧持续时间(ms)
    """
    # 收集所有step的文件
    ct_files = sorted([f for f in os.listdir(ct_steps_dir) if f.startswith('combined_img_step_')])
    mask_files = sorted([f for f in os.listdir(mask_steps_dir) if f.startswith('combined_step_')])
    
    rgb_frames = []
    total_steps = len(ct_files)
    
    for i, (ct_file, mask_file) in enumerate(zip(ct_files, mask_files)):
        # 读取当前step的CT和mask
        ct_path = os.path.join(ct_steps_dir, ct_file)
        mask_path = os.path.join(mask_steps_dir, mask_file)
        
        ct = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        
        ct_array = sitk.GetArrayFromImage(ct)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # 找到肿瘤最大的切片
        max_slice_idx = find_largest_tumor_slice(ct_array, mask_array, axis)
        
        # 获取对应的CT和mask切片
        ct_slice, mask_slice = get_slice_from_volume(ct_array, mask_array, max_slice_idx, axis)
        
        # 创建并排显示的图像
        combined_img = create_side_by_side_view(ct_slice, mask_slice)
        
        # 添加方向和步骤信息
        direction_text = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[axis]
        cv2.putText(combined_img,
                   f'{direction_text} Step {i*5:03d}/{total_steps*5-5:03d}',
                   (50, 50),
                   cv2.FONT_HERSHEY_COMPLEX,
                   1,
                   (255, 255, 255),
                   2)
        
        rgb_frames.append(combined_img)
    
    # 转换为PIL图像并保存为GIF
    pil_frames = [Image.fromarray(frame) for frame in rgb_frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

def main():
    # 设置输入输出路径
    ct_steps_dir = "/opt/data/private/wny/tumor_visualization/output/combined_img"
    mask_steps_dir = "/opt/data/private/wny/tumor_visualization/output/combined_mask"
    output_dir = "/opt/data/private/wny/tumor_visualization/output/gifs"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成三个方向的肿瘤生长过程GIF
    directions = ['sagittal', 'coronal', 'axial']
    for i, direction in enumerate(directions):
        output_path = os.path.join(output_dir, f"tumor_growth_largest_{direction}.gif")
        create_tumor_growth_gif(ct_steps_dir, mask_steps_dir, output_path, axis=i)
        print(f"Generated tumor growth GIF for {direction} view: {output_path}")

if __name__ == "__main__":
    main()
