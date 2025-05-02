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
        6: {"name": "bile_duct", "desc": "Bile Duct", "color": (0.0, 1.0, 1.0), "opacity": 0.8},                              # Cyan
        7: {"name": "pancreatic_cyst", "desc": "Pancreatic Cyst", "color": (0.8, 0.6, 1.0), "opacity": 0.7},                  # Light Purple
        8: {"name": "pancreatic_pnet", "desc": "Pancreatic Neuroendocrine Tumor", "color": (1.0, 0.6, 0.2), "opacity": 0.9},  # Orange
        9: {"name": "postcava", "desc": "Postcava", "color": (0.4, 0.0, 0.8), "opacity": 0.8}                                 # Dark Purple
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
    """Capture 3D view directly from NII file"""
    # Read NIFTI file and get label information
    itk_img = itk.imread(nii_path)
    numpy_img = itk.array_from_image(itk_img)
    unique_labels = np.unique(numpy_img)
    
    # If first run, show available labels and get user selection
    if selected_labels is None:
        available_labels = show_available_labels(unique_labels)
        selected_labels = get_user_label_selection(available_labels)
    
    vtk_img = itk.vtk_image_from_image(itk_img)

    # Create renderer and enable transparency support
    renderer = vtkRenderer()
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(*window_size)
    render_window.SetOffScreenRendering(1)
    renderer.SetBackground(0, 0, 0)
    
    # Enable transparency rendering
    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)
    render_window.SetAlphaBitPlanes(1)
    render_window.SetMultiSamples(0)

    # Add lighting
    light = vtk.vtkLight()
    light.SetIntensity(1.5)
    light.SetPosition(0, 0, 100)
    renderer.AddLight(light)
    
    # Backlight
    back_light = vtk.vtkLight()
    back_light.SetIntensity(0.8)
    back_light.SetPosition(0, 0, -100)
    renderer.AddLight(back_light)

    # Initialize variables
    bounds = [float('inf'), float('-inf'), 
             float('inf'), float('-inf'),
             float('inf'), float('-inf')]  # Modified initial boundary values
    actors = []  # Initialize actors list
    
    # Only process user-selected labels
    label_props = get_label_properties()
    rendered_labels = []
    
    for label_value in selected_labels:
        if label_value not in unique_labels:
            print(f"Skipping label {label_value}: Not found in data")
            continue
            
        mesh = process_nii_to_mesh(vtk_img, label_value)
        if not mesh.GetNumberOfPoints():
            print(f"Skipping label {label_value}: No valid mesh generated")
            continue
            
        rendered_labels.append(label_value)
        props = label_props[label_value]
        
        print(f"Setting label {label_value} ({props['name']}) color to: {props['color']}, opacity: {props['opacity']}")
        
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        
        # Set color and opacity
        color = props["color"]
        opacity = props["opacity"]
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetOpacity(opacity)
        
        # Set material properties
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(50)
        
        actors.append((actor, props["name"]))
        print(f"Successfully created actor for {props['name']}, color: R={color[0]}, G={color[1]}, B={color[2]}, opacity: {opacity}")
        
        # Update overall bounds
        actor_bounds = actor.GetBounds()
        for i in range(6):
            if i % 2 == 0:  # min values
                bounds[i] = min(bounds[i], actor_bounds[i])
            else:  # max values
                bounds[i] = max(bounds[i], actor_bounds[i])

    # Check if there are valid bounds
    if bounds[0] == float('inf'):
        print("Warning: No valid meshes found!")
        return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

    # Calculate appropriate scaling and center
    center = [(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, (bounds[5]+bounds[4])/2]
    scale = 1.0 / max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    
    # Second pass: apply scaling and add to renderer
    for actor, name in actors:
        # Reset position and apply scaling
        actor.SetPosition(-center[0], -center[1], -center[2])
        actor.SetScale(scale)
        renderer.AddActor(actor)
        print(f"Adding {name} to renderer")

    # Set camera
    camera = vtkCamera()
    camera = adjust_camera_settings(camera, angle_h, angle_v)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()

    # Render and capture image
    render_window.Render()
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.Update()

    # Convert to numpy array
    vtk_image = window_to_image.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    numpy_array = vtk_to_numpy(vtk_array).reshape(height, width, components)
    
    print("\nRendering Results Statistics:")
    print(f"Successfully rendered labels: {rendered_labels}")
    
    return numpy_array, selected_labels

def create_growth_animation(nii_base_dir, output_path, angle_h=0, angle_v=0, duration=500, reverse_order=False):
    """Create tumor growth animation directly from NII files
    
    Args:
        nii_base_dir: NII files directory
        output_path: Output GIF file path
        angle_h: Horizontal view angle
        angle_v: Vertical view angle
        duration: Duration of each GIF frame (milliseconds)
        reverse_order: Whether to reverse the order (large to small)
    """
    # Find all nii.gz files
    nii_files = [f for f in os.listdir(nii_base_dir) if f.endswith('.nii.gz')]
    
    if not nii_files:
        print("No NII files found!")
        return
    
    # Sort files
    # 1. Find step files
    step_files = [f for f in nii_files if 'step_' in f]
    # 2. Extract step numbers
    step_numbers = {}
    for f in step_files:
        try:
            # Extract step number from filename
            step_str = f.split('step_')[1].split('.')[0]
            step_num = int(step_str)
            step_numbers[f] = step_num
        except (IndexError, ValueError):
            # Skip if unable to extract number
            continue
    
    # 3. Sort by step number
    sorted_files = sorted(step_files, key=lambda f: step_numbers.get(f, float('inf')))
    
    # 4. If there are final files, add them at the end
    final_files = [f for f in nii_files if 'final' in f]
    sorted_files.extend(final_files)
    
    # 5. If reverse order is requested, reverse the file list
    if reverse_order:
        sorted_files.reverse()
        order_desc = "large to small (reversed)"
    else:
        order_desc = "small to large (forward)"
    
    print(f"Processing {len(sorted_files)} files, sorted by growth step from {order_desc}")
    
    frames = []
    selected_labels = None  # Will prompt user on first run
    
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
    nii_base_dir = "E:\zhuomian\Synthetic_Tumor_visualization\labels"
    output_dir = "E:\zhuomian\Synthetic_Tumor_visualization/3Dgif"
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
