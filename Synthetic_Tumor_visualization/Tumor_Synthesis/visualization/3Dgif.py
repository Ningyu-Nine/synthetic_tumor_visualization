import numpy as np
import vtk
import itk
from PIL import Image
import os
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor,
    vtkPolyDataMapper, vtkActor, vtkCamera
)
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
from vtkmodules.util.numpy_support import vtk_to_numpy

def get_label_properties():
    """Get label properties: colors and opacities"""
    return {
        1: {"name": "PDAC_lesion", "desc": "Pancreatic Ductal Adenocarcinoma", "color": (1.0, 0.0, 0.0), "opacity": 1.0},    # Red
        2: {"name": "Veins", "desc": "Veins", "color": (0.0, 0.0, 1.0), "opacity": 0.8},                                      # Blue
        3: {"name": "Arteries", "desc": "Arteries", "color": (1.0, 0.4, 0.4), "opacity": 0.8},                                # Pink
        4: {"name": "pancreas", "desc": "Pancreatic Parenchyma", "color": (0.2, 0.8, 0.2), "opacity": 0.4},                   # Green
        5: {"name": "pancreatic_duct", "desc": "Pancreatic Duct", "color": (1.0, 1.0, 0.0), "opacity": 0.8},                  # Yellow
        6: {"name": "bile_duct", "desc": "Bile Duct", "color": (0.0, 1.0, 1.0), "opacity": 0.8}                               # Cyan
    }

def show_available_labels(unique_labels):
    """Display available label information"""
    label_props = get_label_properties()
    print("\nAvailable labels in current image:")
    available_labels = []
    for label in unique_labels:
        if label != 0 and label in label_props:  # Exclude background
            props = label_props[label]
            print(f"[{int(label)}] {props['desc']} ({props['name']})")
            available_labels.append(int(label))
    return available_labels

def get_user_label_selection(available_labels):
    """Get user's label selection"""
    while True:
        print("\nSelect labels to display (enter numbers separated by commas, e.g.: 2,3,4):")
        try:
            selection = input().strip()
            if selection.lower() == 'all':
                return available_labels
            
            selected_labels = [int(x.strip()) for x in selection.split(',')]
            valid_labels = [x for x in selected_labels if x in available_labels]
            
            if not valid_labels:
                print("No valid labels selected, please try again")
                continue
                
            return valid_labels
        except ValueError:
            print("Invalid input format, please try again")

def process_nii_to_mesh(vtk_img, label_value, smoothing_iterations=25):
    """Generate 3D mesh from NII data"""
    # Extract isosurface
    contour = vtkDiscreteMarchingCubes()
    contour.SetInputData(vtk_img)
    contour.GenerateValues(1, label_value, label_value)
    contour.Update()

    # Smooth
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    return smoother.GetOutput()

def adjust_camera_settings(camera, angle_h, angle_v):
    """Adjust camera settings to avoid viewing angle issues"""
    # Base camera distance
    distance = 2.0
    
    # Convert angles to radians
    h_rad = np.radians(angle_h)
    v_rad = np.radians(angle_v)
    
    # Calculate camera position
    x = distance * np.sin(h_rad) * np.cos(v_rad)
    y = distance * np.sin(v_rad)
    z = distance * np.cos(h_rad) * np.cos(v_rad)
    
    camera.SetPosition(x, y, z)
    camera.SetFocalPoint(0, 0, 0)
    
    # Intelligent view-up vector setting
    if abs(angle_v) > 80:  # Near vertical view
        up_x = -np.sin(h_rad)
        up_y = 0
        up_z = -np.cos(h_rad)
        camera.SetViewUp(up_x, up_y, up_z)
    else:  # Normal view
        camera.SetViewUp(0, 1, 0)
    
    camera.SetParallelProjection(True)
    camera.SetParallelScale(1.0)
    
    return camera

def capture_nii_3d_view(nii_path, angle_h, angle_v, selected_labels=None, window_size=(800, 800)):
    """直接从NII文件捕获3D视图"""
    # 读取NIFTI文件并获取标签信息
    itk_img = itk.imread(nii_path)
    numpy_img = itk.array_from_image(itk_img)
    unique_labels = np.unique(numpy_img)
    
    # 如果是第一次运行，显示可用标签并获取用户选择
    if selected_labels is None:
        available_labels = show_available_labels(unique_labels)
        selected_labels = get_user_label_selection(available_labels)
    
    vtk_img = itk.vtk_image_from_image(itk_img)

    # 创建渲染器并启用透明度支持
    renderer = vtkRenderer()
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(*window_size)
    render_window.SetOffScreenRendering(1)
    renderer.SetBackground(0, 0, 0)
    
    # 启用透明度渲染
    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)
    render_window.SetAlphaBitPlanes(1)
    render_window.SetMultiSamples(0)

    # 添加光照
    light = vtk.vtkLight()
    light.SetIntensity(1.5)
    light.SetPosition(0, 0, 100)
    renderer.AddLight(light)
    
    # 背光
    back_light = vtk.vtkLight()
    back_light.SetIntensity(0.8)
    back_light.SetPosition(0, 0, -100)
    renderer.AddLight(back_light)

    # 初始化变量
    bounds = [float('inf'), float('-inf'), 
             float('inf'), float('-inf'),
             float('inf'), float('-inf')]  # 修改初始边界值
    actors = []  # 初始化actors列表
    
    # 只处理用户选择的标签
    label_props = get_label_properties()
    rendered_labels = []
    
    for label_value in selected_labels:
        if label_value not in unique_labels:
            print(f"跳过标签 {label_value}: 在数据中未找到")
            continue
            
        mesh = process_nii_to_mesh(vtk_img, label_value)
        if not mesh.GetNumberOfPoints():
            print(f"跳过标签 {label_value}: 未生成有效网格")
            continue
            
        rendered_labels.append(label_value)
        props = label_props[label_value]
        
        print(f"设置标签 {label_value} ({props['name']}) 的颜色为: {props['color']}, 透明度: {props['opacity']}")
        
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        
        # 设置颜色和透明度
        color = props["color"]
        opacity = props["opacity"]
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetOpacity(opacity)
        
        # 设置材质属性
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(50)
        
        actors.append((actor, props["name"]))
        print(f"成功创建 {props['name']} 的actor，颜色为: R={color[0]}, G={color[1]}, B={color[2]}, 透明度: {opacity}")
        
        # 更新总体边界
        actor_bounds = actor.GetBounds()
        for i in range(6):
            if i % 2 == 0:  # min values
                bounds[i] = min(bounds[i], actor_bounds[i])
            else:  # max values
                bounds[i] = max(bounds[i], actor_bounds[i])

    # 检查是否有有效的边界
    if bounds[0] == float('inf'):
        print("警告: 没有找到有效的网格！")
        return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

    # 计算合适的缩放和中心
    center = [(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, (bounds[5]+bounds[4])/2]
    scale = 1.0 / max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    
    # 第二遍：应用缩放并添加到渲染器
    for actor, name in actors:
        # 重置位置并应用缩放
        actor.SetPosition(-center[0], -center[1], -center[2])
        actor.SetScale(scale)
        renderer.AddActor(actor)
        print(f"Adding {name} to renderer")

    # 设置相机
    camera = vtkCamera()
    camera = adjust_camera_settings(camera, angle_h, angle_v)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()

    # 渲染并捕获图像
    render_window.Render()
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.Update()

    # 转换为numpy数组
    vtk_image = window_to_image.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    numpy_array = vtk_to_numpy(vtk_array).reshape(height, width, components)
    
    print("\n渲染结果统计:")
    print(f"成功渲染的标签: {rendered_labels}")
    
    return numpy_array, selected_labels

def create_growth_animation(nii_base_dir, output_path, angle_h=0, angle_v=0, duration=500, reverse_order=False):
    """直接从NII文件创建肿瘤生长动画
    
    Args:
        nii_base_dir: NII文件目录
        output_path: 输出GIF文件路径
        angle_h: 水平视角角度
        angle_v: 垂直视角角度
        duration: 每帧GIF持续时间(毫秒)
        reverse_order: 是否逆序排列(从大到小)
    """
    # 查找所有nii.gz文件
    nii_files = [f for f in os.listdir(nii_base_dir) if f.endswith('.nii.gz')]
    
    if not nii_files:
        print("No NII files found!")
        return
    
    # 对文件进行排序
    # 1. 找出step文件
    step_files = [f for f in nii_files if 'step_' in f]
    # 2. 提取步骤编号
    step_numbers = {}
    for f in step_files:
        try:
            # 从文件名中提取步骤编号
            step_str = f.split('step_')[1].split('.')[0]
            step_num = int(step_str)
            step_numbers[f] = step_num
        except (IndexError, ValueError):
            # 如果无法提取编号，则跳过
            continue
    
    # 3. 按步骤编号排序
    sorted_files = sorted(step_files, key=lambda f: step_numbers.get(f, float('inf')))
    
    # 4. 如果有final文件，将其放在最后
    final_files = [f for f in nii_files if 'final' in f]
    sorted_files.extend(final_files)
    
    # 5. 如果需要逆序，反转文件列表
    if reverse_order:
        sorted_files.reverse()
        order_desc = "从大到小(逆序)"
    else:
        order_desc = "从小到大(正序)"
    
    print(f"处理 {len(sorted_files)} 个文件，按照生长步骤{order_desc}排序")
    
    frames = []
    selected_labels = None  # 第一次运行时会提示用户选择
    
    for nii_file in sorted_files:
        file_path = os.path.join(nii_base_dir, nii_file)
        print(f"Processing {nii_file}...")
        
        image_array, selected_labels = capture_nii_3d_view(
            file_path, angle_h, angle_v, selected_labels=selected_labels
        )
        image = Image.fromarray(image_array.astype('uint8'))
        frames.append(image)
    
    if frames:
        print(f"Saving GIF with {len(frames)} frames...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )

def get_user_order_preference():
    """Get user's order preference"""
    while True:
        print("\nSelect GIF animation order:")
        print("1. Forward - Show tumor growth from small to large")
        print("2. Reverse - Show tumor shrinkage from large to small")
        try:
            choice = input("Enter option (1 or 2): ").strip()
            if choice == '1':
                return False  # Forward
            elif choice == '2':
                return True   # Reverse
            else:
                print("Invalid choice, please enter 1 or 2")
        except Exception:
            print("Input error, please try again")

def get_user_view():
    """Get user input for view parameters"""
    print("\nEnter view parameters:")
    print("Horizontal angle (Azimuth): -180 to 180 degrees, 0° is front view, 90° is right side")
    print("Vertical angle (Elevation): -90 to 90 degrees, 0° is horizontal view, 90° is top view")
    
    while True:
        try:
            angle_h = float(input("Enter horizontal angle: "))
            if -180 <= angle_h <= 180:
                break
            print("Horizontal angle must be between -180 and 180 degrees")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            angle_v = float(input("Enter vertical angle: "))
            if -90 <= angle_v <= 90:
                break
            print("Vertical angle must be between -90 and 90 degrees")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        view_name = input("Name this view (e.g.: front_45deg): ").strip()
        if view_name:
            break
        print("View name cannot be empty")
    
    return angle_h, angle_v, view_name

def main():
    # Update base path for NII files
    nii_base_dir = "/opt/data/private/wny/tumor_visualization/output/growth_process/labels"
    output_dir = "/opt/data/private/wny/tumor_visualization/output/3Dgif"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nReading tumor growth steps from directory: {nii_base_dir}")
    
    # Get user preference for order
    reverse_order = get_user_order_preference()
    
    if reverse_order:
        print("Generating GIF in reverse order (large to small), showing tumor shrinkage")
    else:
        print("Generating GIF in forward order (small to large), showing tumor growth")

    while True:
        angle_h, angle_v, view_name = get_user_view()
        
        # Adjust output filename based on order
        if reverse_order:
            output_filename = f"tumor_shrink_{view_name}.gif"
        else:
            output_filename = f"tumor_growth_{view_name}.gif"
            
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nGenerating animation for {view_name} view...")
        create_growth_animation(
            nii_base_dir, 
            output_path, 
            angle_h=angle_h, 
            angle_v=angle_v,
            duration=500,
            reverse_order=reverse_order
        )
        print(f"Saved to: {output_path}")
        
        choice = input("\nAdd another view? (y/n): ").lower()
        if choice != 'y':
            break
    
    print("\nAll view animations completed!")

if __name__ == "__main__":
    main()
