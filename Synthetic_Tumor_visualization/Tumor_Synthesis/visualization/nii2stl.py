import itk
import os
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter, vtkAppendPolyData
from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, 
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor)
from vtkmodules.vtkIOGeometry import vtkSTLWriter

def process_and_save_mesh(vtk_img, label_value, color_name, output_path, smoothing_iterations=25):
    """处理单个标签并保存为STL"""
    # 提取等值面
    contour = vtkDiscreteMarchingCubes()
    contour.SetInputData(vtk_img)
    contour.GenerateValues(1, label_value, label_value)
    contour.Update()

    # 平滑处理
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # 保存STL
    writer = vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_path)
    writer.Write()

    return smoother.GetOutput()

def process_and_save_combined_mesh(vtk_img, output_path, smoothing_iterations=25):
    """处理并保存合并的肿瘤和胰腺模型"""
    # 创建合并器
    append = vtkAppendPolyData()
    
    # 处理胰腺(label=2)
    contour_pancreas = vtkDiscreteMarchingCubes()
    contour_pancreas.SetInputData(vtk_img)
    contour_pancreas.GenerateValues(1, 2, 2)
    contour_pancreas.Update()

    # 处理肿瘤(label=1)
    contour_tumor = vtkDiscreteMarchingCubes()
    contour_tumor.SetInputData(vtk_img)
    contour_tumor.GenerateValues(1, 1, 1)
    contour_tumor.Update()

    # 合并两个模型
    append.AddInputConnection(contour_pancreas.GetOutputPort())
    append.AddInputConnection(contour_tumor.GetOutputPort())
    append.Update()

    # 平滑处理
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(append.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # 保存合并的STL
    writer = vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_path)
    writer.Write()

    return smoother.GetOutput()

def show_3d_mask(nifti_file_name, output_dir, show_3d=True):
    """显示并保存3D模型"""
    if not os.path.exists(nifti_file_name):
        raise FileNotFoundError(f"File not found: {nifti_file_name}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing file: {nifti_file_name}")
    
    # 读取NIFTI文件
    itk_img = itk.imread(filename=nifti_file_name)
    vtk_img = itk.vtk_image_from_image(itk_img)

    # 设置渲染器（如果需要显示）
    if show_3d:
        colors = vtkNamedColors()
        renderer = vtkRenderer()
        renderer.SetBackground(colors.GetColor3d("Black"))

    # 处理每个标签
    label_colors = [(1, "Red", "tumor"), (2, "Green", "pancreas")]
    poly_data_dict = {}
    
    for label_value, color_name, label_name in label_colors:
        # 生成输出文件路径
        output_path = os.path.join(output_dir, f"{label_name}.stl")
        
        # 处理并保存mesh
        poly_data = process_and_save_mesh(vtk_img, label_value, color_name, output_path)
        poly_data_dict[label_name] = poly_data
        
        if show_3d:
            # 创建actor用于显示
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            mapper.ScalarVisibilityOff()

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors.GetColor3d(color_name))
            renderer.AddActor(actor)

    # 添加保存合并模型的代码
    combined_output_path = os.path.join(output_dir, "combined_tumor_pancreas.stl")
    print(f"Saving combined model to: {combined_output_path}")
    process_and_save_combined_mesh(vtk_img, combined_output_path)

    # 如果需要显示3D视图
    if show_3d:
        render_window = vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        interactor = vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.Initialize()
        render_window.Render()
        interactor.Start()

    return poly_data_dict

def process_directory(input_dir, output_base_dir, show_3d=False):
    """处理目录中的所有 .nii.gz 文件"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # 获取所有 .nii.gz 文件
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    for nii_file in nii_files:
        # 获取文件名（不带扩展名）用作子目录名
        subfolder_name = os.path.splitext(os.path.splitext(nii_file)[0])[0]
        output_dir = os.path.join(output_base_dir, subfolder_name)
        
        # 创建子目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理当前文件
        nii_path = os.path.join(input_dir, nii_file)
        print(f"\nProcessing: {nii_file}")
        try:
            show_3d_mask(nii_path, output_dir, show_3d=show_3d)
            print(f"STL files saved to {output_dir}")
        except Exception as e:
            print(f"Error processing {nii_file}: {e}")

def main():
    # 更新输入输出路径
    input_dir = "/opt/data/private/wny/tumor_visualization/output/combined_mask"
    output_base_dir = "/opt/data/private/wny/tumor_visualization/output/stl"
    
    try:
        process_directory(input_dir, output_base_dir, show_3d=False)
        print("\nAll files processed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == '__main__':
    main()
