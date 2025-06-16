import numpy as np
import SimpleITK as sitk
import cv2
from PIL import Image
import os
import argparse
from skimage import measure

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate 2D tumor growth animations')
    parser.add_argument('--input-case', type=str, 
                      default='BDMAP_A0001000',
                      help='Case ID to visualize (default: BDMAP_A0001000)')
    parser.add_argument('--output-root', type=str,
                      default='../../output',
                      help='Root directory for outputs')
    parser.add_argument('--growth-process-root', type=str,
                      default='../../output/growth_process',
                      help='Root directory containing growth process data')
    parser.add_argument('--duration', type=int, default=500,
                      help='Duration of each frame in milliseconds (default: 500)')
    return parser.parse_args()

def get_paths(args):
    """Generate standardized paths"""
    # Input paths
    case_dir = os.path.join(args.growth_process_root, args.input_case)
    images_dir = os.path.join(case_dir, 'images')
    labels_dir = os.path.join(case_dir, 'labels')
    
    # Output paths
    output_dir = os.path.join(args.output_root, 'gifs', args.input_case)
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'output_dir': output_dir,
        'case_id': args.input_case
    }

def validate_paths(paths):
    """Validate input paths exist"""
    if not os.path.exists(paths['images_dir']):
        raise FileNotFoundError(f"Images directory not found: {paths['images_dir']}")
    if not os.path.exists(paths['labels_dir']):
        raise FileNotFoundError(f"Labels directory not found: {paths['labels_dir']}")
    
    # Check for step files
    ct_files = [f for f in os.listdir(paths['images_dir']) if f.startswith('tumor_step_')]
    mask_files = [f for f in os.listdir(paths['labels_dir']) if f.startswith('tumor_step_')]
    
    if not ct_files or not mask_files:
        raise FileNotFoundError("No tumor step files found in input directories")
    
    print(f"Input validation passed:")
    print(f"  Images directory: {paths['images_dir']} ({len(ct_files)} files)")
    print(f"  Labels directory: {paths['labels_dir']} ({len(mask_files)} files)")

def find_largest_tumor_slice(ct_array, mask_array, axis=0):
    """Find the slice with the largest tumor area
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
    """Extract slice from volume according to specified direction"""
    if axis == 0:
        return volume[slice_idx], mask[slice_idx]
    elif axis == 1:
        return volume[:, slice_idx, :], mask[:, slice_idx, :]
    else:  # axis == 2
        return volume[:, :, slice_idx], mask[:, :, slice_idx]

def get_mask_contours(mask_slice):
    """Get contours from mask"""
    # Use OpenCV contour detection instead of skimage
    mask_slice = mask_slice.astype(np.uint8)
    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_side_by_side_view(ct_slice, mask_slice, window_center=40, window_width=400):
    """Create side-by-side view of original CT and annotated CT"""
    # CT windowing
    ct_min = window_center - window_width/2
    ct_max = window_center + window_width/2
    ct_slice = np.clip(ct_slice, ct_min, ct_max)
    normalized = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
    
    # Create two RGB images
    original_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    annotated_rgb = original_rgb.copy()
    
    # Get pancreas and tumor contours separately
    pancreas_binary = (mask_slice == 2).astype(np.uint8)
    tumor_binary = (mask_slice == 1).astype(np.uint8)
    
    # Get contours
    pancreas_contours, _ = cv2.findContours(pancreas_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tumor_contours, _ = cv2.findContours(tumor_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    cv2.drawContours(annotated_rgb, pancreas_contours, -1, (0, 255, 0), 1)  # Green contour for pancreas
    cv2.drawContours(annotated_rgb, tumor_contours, -1, (255, 0, 0), 1)  # Red contour for tumor
    
    # Create side-by-side image
    combined_img = np.hstack([original_rgb, annotated_rgb])
    return combined_img

def create_tumor_growth_gif(images_dir, labels_dir, output_path, axis=0, duration=500):
    """
    Create GIF showing tumor growth process
    Parameters:
        images_dir: CT images directory
        labels_dir: mask directory
        output_path: Output GIF path
        axis: 0 for sagittal, 1 for coronal, 2 for axial
        duration: Frame duration in milliseconds
    """
    # Collect all step files
    ct_files = sorted([f for f in os.listdir(images_dir) if f.startswith('tumor_step_')])
    mask_files = sorted([f for f in os.listdir(labels_dir) if f.startswith('tumor_step_')])
    
    if len(ct_files) != len(mask_files):
        print(f"Warning: Mismatch in file counts - CT: {len(ct_files)}, Masks: {len(mask_files)}")
        # Use the minimum count
        min_count = min(len(ct_files), len(mask_files))
        ct_files = ct_files[:min_count]
        mask_files = mask_files[:min_count]
    
    rgb_frames = []
    total_steps = len(ct_files)
    
    for i, (ct_file, mask_file) in enumerate(zip(ct_files, mask_files)):
        # Read current step CT and mask
        ct_path = os.path.join(images_dir, ct_file)
        mask_path = os.path.join(labels_dir, mask_file)
        
        ct = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        
        ct_array = sitk.GetArrayFromImage(ct)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Find slice with largest tumor
        max_slice_idx = find_largest_tumor_slice(ct_array, mask_array, axis)
        
        # Get corresponding CT and mask slices
        ct_slice, mask_slice = get_slice_from_volume(ct_array, mask_array, max_slice_idx, axis)
        
        # Create side-by-side view
        combined_img = create_side_by_side_view(ct_slice, mask_slice)
        
        # Add direction and step information
        direction_text = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[axis]
        
        # Extract step number from filename
        try:
            step_num = int(ct_file.split('step_')[1].split('.')[0])
        except:
            step_num = i
            
        cv2.putText(combined_img,
                   f'{direction_text} Step {step_num:03d}',
                   (50, 50),
                   cv2.FONT_HERSHEY_COMPLEX,
                   1,
                   (255, 255, 255),
                   2)
        
        rgb_frames.append(combined_img)
    
    # Convert to PIL images and save as GIF
    pil_frames = [Image.fromarray(frame) for frame in rgb_frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

def main():
    args = parse_args()
    
    # Get and validate paths
    paths = get_paths(args)
    validate_paths(paths)
    
    print(f"\nGenerating 2D slice animations for case: {args.input_case}")
    print(f"Input directories:")
    print(f"  Images: {paths['images_dir']}")
    print(f"  Labels: {paths['labels_dir']}")
    print(f"Output directory: {paths['output_dir']}")
    
    # Generate tumor growth GIFs for three directions
    directions = ['sagittal', 'coronal', 'axial']
    for i, direction in enumerate(directions):
        output_filename = f"{args.input_case}_tumor_growth_largest_{direction}.gif"
        output_path = os.path.join(paths['output_dir'], output_filename)
        
        print(f"\nGenerating {direction} view animation...")
        create_tumor_growth_gif(
            paths['images_dir'], 
            paths['labels_dir'], 
            output_path, 
            axis=i,
            duration=args.duration
        )
        print(f"Generated: {output_path}")
    
    print(f"\nAll 2D animations for {args.input_case} completed!")
    print(f"Output directory: {paths['output_dir']}")

if __name__ == "__main__":
    main()
