import os
import shutil
import numpy as np
import SimpleITK as sitk

def merge_segmentations(input_dir, output_label_path, output_image_path):
    """
    Merge multiple individual NII segmentation files into a unified label file.
    
    Args:
        input_dir: Directory containing individual organ segmentation NII files
        output_label_path: Path to save the merged label file
        output_image_path: Path to save the CT image file
    """
    # Define segmentation files and corresponding label values
    segmentation_map = {
        "pancreatic_pdac.nii.gz": 1,     # PDAC lesion (red)
        "veins.nii.gz": 2,               # Veins (blue)
        "aorta.nii.gz": 3,               # Arteries - aorta (pink)
        "celiac_aa.nii.gz": 3,           # Arteries - celiac artery (pink)
        "superior_mesenteric_artery.nii.gz": 3,  # Arteries - superior mesenteric artery (pink)
        "pancreas.nii.gz": 4,            # Pancreas (green)
        "pancreatic_duct.nii.gz": 5,     # Pancreatic duct (yellow)
        "common_bile_duct.nii.gz": 6,    # Bile duct (cyan)
        "pancreatic_cyst.nii.gz": 7,     # Pancreatic cyst (new label)
        "pancreatic_pnet.nii.gz": 8,     # Pancreatic pnet (new label)
        "postcava.nii.gz": 9             # Postcava (new label)
    }
    
    # Get the target CT image path
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
    
    # Create blank label image, same size as CT image
    print("Creating blank label image")
    label_array = np.zeros(sitk.GetArrayFromImage(ct_image).shape, dtype=np.uint8)
    
    # Process each segmentation file
    seg_files_found = []
    for seg_file, label_value in segmentation_map.items():
        seg_path = os.path.join(input_dir, seg_file)
        if os.path.exists(seg_path):
            seg_files_found.append(seg_file)
            print(f"Processing {seg_file} (label value: {label_value})")
            
            # Read segmentation data
            seg_image = sitk.ReadImage(seg_path)
            seg_array = sitk.GetArrayFromImage(seg_image)
            
            # Set nonzero values to the corresponding label value
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
    print("Done!")

def main():
    # Input and output paths
    base_dir = "/opt/data/private/wny/Synthetic_Tumor_visualization"
    input_seg_dir = os.path.join(base_dir, "subset_AbdomenAtlasF/BDMAP_A0001000/segmentations")
    
    # Output paths
    output_label_path = os.path.join(base_dir, "label/merged_labels.nii.gz")
    output_image_path = os.path.join(base_dir, "image/ct.nii.gz")
    
    # Merge segmentations
    merge_segmentations(input_seg_dir, output_label_path, output_image_path)
    
    # Show label mapping for 3Dgif.py
    print("\nLabel mapping for 3Dgif.py:")
    print("1: PDAC lesion (red)")
    print("2: Veins (blue)")
    print("3: Arteries (pink)")
    print("4: Pancreas parenchyma (green)")
    print("5: Pancreatic duct (yellow)")
    print("6: Bile duct (cyan)")
    print("7: Pancreatic cyst (new)")
    print("8: Pancreatic pnet (new)")
    print("9: Postcava (new)")
    
    print("\nYou can now run 3Dgif.py to visualize the merged labels.")

if __name__ == "__main__":
    main()
