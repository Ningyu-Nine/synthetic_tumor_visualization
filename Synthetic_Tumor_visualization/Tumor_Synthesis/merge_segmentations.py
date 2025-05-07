import os
import shutil
import numpy as np
import SimpleITK as sitk

def merge_segmentations(input_dir, output_label_path, output_image_path):
    """
    Merge multiple separate NII segmentation files into a unified label file
    
    Args:
        input_dir: Directory containing organ segmentation NII files
        output_label_path: Path to save the merged label file
        output_image_path: Path to save the CT image file
    """
    # Define segmentation files and corresponding label values
    segmentation_map = {
        "pancreatic_pdac.nii.gz": 1,     # PDAC lesion (Red)
        "veins.nii.gz": 2,               # Veins (Blue)
        "aorta.nii.gz": 3,               # Arteries - Aorta (Pink)
        "celiac_aa.nii.gz": 3,           # Arteries - Celiac artery (Pink)
        "superior_mesenteric_artery.nii.gz": 10,  # Superior mesenteric artery (Gold) - Separate display
        "pancreas.nii.gz": 4,            # Pancreas (Green)
        "pancreatic_duct.nii.gz": 5,     # Pancreatic duct (Yellow)
        "common_bile_duct.nii.gz": 6,    # Bile duct (Cyan)
        "pancreatic_cyst.nii.gz": 7,     # Pancreatic cyst
        "pancreatic_pnet.nii.gz": 8,     # Pancreatic neuroendocrine tumor
        "postcava.nii.gz": 9             # Postcava
    }
    
    # Get target CT image path
    ct_path = os.path.join(os.path.dirname(input_dir), "ct.nii.gz")
    
    # Create output directories
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # First load CT image to get image properties
    print(f"Reading CT image: {ct_path}")
    ct_image = sitk.ReadImage(ct_path)
    
    # Copy CT image to target location
    print(f"Copying CT image to: {output_image_path}")
    shutil.copy2(ct_path, output_image_path)
    
    # Create empty label image with same size as CT image
    print("Creating empty label image")
    label_array = np.zeros(sitk.GetArrayFromImage(ct_image).shape, dtype=np.uint8)
    
    # Process each segmentation file
    seg_files_found = []
    for seg_file, label_value in segmentation_map.items():
        seg_path = os.path.join(input_dir, seg_file)
        if os.path.exists(seg_path):
            seg_files_found.append(seg_file)
            print(f"Processing {seg_file} (Label value: {label_value})")
            
            # Read segmentation data
            seg_image = sitk.ReadImage(seg_path)
            seg_array = sitk.GetArrayFromImage(seg_image)
            
            # Set non-zero values to corresponding label value
            label_array[seg_array > 0] = label_value
        else:
            print(f"Segmentation file not found: {seg_path}")
    
    print(f"Successfully processed segmentation files: {len(seg_files_found)}/{len(segmentation_map)}")
    
    # Convert back to SimpleITK image
    merged_label_image = sitk.GetImageFromArray(label_array)
    merged_label_image.CopyInformation(ct_image)  # Copy metadata from original CT image
    
    # Save merged label image
    print(f"Saving merged label file to: {output_label_path}")
    sitk.WriteImage(merged_label_image, output_label_path)
    print("Complete!")

def main():
    # Input and output paths
    base_dir = "/opt/data/private/wny/Synthetic_Tumor_visualization"
    input_seg_dir = os.path.join(base_dir, "subset_AbdomenAtlasF/BDMAP_A0001000/segmentations")
    
    # Output paths
    output_label_path = os.path.join(base_dir, "label/merged_labels.nii.gz")
    output_image_path = os.path.join(base_dir, "image/ct.nii.gz")
    
    # Merge segmentations
    merge_segmentations(input_seg_dir, output_label_path, output_image_path)
    
    # Show available label mapping for 3Dgif.py
    print("\nAvailable label mapping for 3Dgif.py:")
    print("1: PDAC lesion (Red)")
    print("2: Veins (Blue)")
    print("3: Arteries (Pink)")
    print("4: Pancreas parenchyma (Green)")
    print("5: Pancreatic duct (Yellow)")
    print("6: Bile duct (Cyan)")
    print("7: Pancreatic cyst")
    print("8: Pancreatic pnet")
    print("9: Postcava")
    print("10: Superior mesenteric artery (Gold)")
    
    print("\nNow you can run 3Dgif.py to visualize the merged labels")

if __name__ == "__main__":
    main()
