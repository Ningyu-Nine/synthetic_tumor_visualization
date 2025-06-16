# Synthetic Tumor Visualization

A comprehensive tool for synthesizing and visualizing tumor growth process in medical imaging, specifically designed for pancreatic tumor visualization using AbdomenAtlasF dataset.

## Features

- Automatic segmentation file merging for AbdomenAtlasF dataset
- Realistic tumor growth simulation with cellular automata
- 3D mesh-based visualization with anatomical structures
- 2D slice-based growth animation
- Multiple viewing angles and transparency settings
- Support for both forward (growth) and reverse (shrinkage) animations

## Workflow Overview

```
1. merge_segmentations.py → Merge individual organ segmentations
2. visualize.py → Generate tumor growth process 
3. visualization/3Dgif.py → Create 3D mesh animations
4. visualization/tumor_gross2D_gif.py → Create 2D slice animations
```

## Directory Structure

```
Synthetic_Tumor_visualization/
├── Tumor_Synthesis/
│   ├── merge_segmentations.py    # Step 1: Merge segmentations
│   ├── visualize.py              # Step 2: Generate tumor growth
│   ├── utils_visual.py           # Utility functions
│   ├── cellular.py               # Cellular automata engine
│   └── visualization/
│       ├── 3Dgif.py             # Step 3: 3D visualization
│       └── tumor_gross2D_gif.py # Step 4: 2D visualization
├── subset_AbdomenAtlasF/         # Input dataset
├── output/
│   ├── growth_process/           # Generated tumor steps
│   │   └── {case_id}/
│   │       ├── images/           # CT images with tumor
│   │       └── labels/           # Label masks
│   ├── 3Dgif/                   # 3D animations
│   │   └── {case_id}/
│   └── gifs/                    # 2D animations
│       └── {case_id}/
└── README.md
```

## Prerequisites

- Python 3.7+
- PyTorch (with CUDA support recommended)
- VTK
- SimpleITK
- NumPy
- OpenCV
- PIL/Pillow
- scikit-image
- scipy

## Installation

```bash
pip install torch torchvision vtk SimpleITK numpy opencv-python pillow scikit-image scipy
```

## Complete Workflow Guide

### Step 1: Merge Segmentation Files

First, merge individual organ segmentation files from AbdomenAtlasF dataset into a unified label file.

```bash
python merge_segmentations.py
```

**What this does:**
- Combines separate NII files for different organs
- Creates unified `merged_labels.nii.gz` file
- Copies CT image to standardized location
- Maps different anatomical structures to specific label values

**Input Structure:**
```
BDMAP_A0001000/
├── ct.nii.gz
└── segmentations/
    ├── pancreatic_pdac.nii.gz
    ├── veins.nii.gz
    ├── aorta.nii.gz
    ├── pancreas.nii.gz
    ├── pancreatic_duct.nii.gz
    └── ... (other organ files)
```

**Output:**
```
BDMAP_A0001000/
├── image/
│   └── ct.nii.gz
└── label/
    └── merged_labels.nii.gz
```

**Label Mapping Created:**
- 1: PDAC lesion (Red)
- 2: Veins (Blue) 
- 3: Arteries (Pink)
- 4: Pancreas (Green)
- 5: Pancreatic duct (Yellow)
- 6: Bile duct (Cyan)
- 7: Pancreatic cyst (Light Purple)
- 8: Pancreatic NET (Orange)
- 9: Postcava (Dark Purple)
- 10: Superior mesenteric artery (Gold)

### Step 2: Generate Tumor Growth Process

Generate synthetic tumor growth simulation using cellular automata.

```bash
# Basic usage with default case
python visualize.py

# Process specific case
python visualize.py --input-case BDMAP_A0001001

# Custom parameters
python visualize.py --input-case BDMAP_A0001000 --steps 50 --tumor-points 2
```

#### Command Line Parameters

**Case Selection:**
- `--input-case`: Case ID to process (default: BDMAP_A0001000)
- `--dataset-root`: Path to AbdomenAtlasF dataset (default: ../subset_AbdomenAtlasF/subset_AbdomenAtlasF)
- `--output-root`: Root output directory (default: ../output)

**Tumor Growth Parameters:**
- `--steps`: Total growth steps (default: 30)
- `--tumor-points`: Initial tumor seed points (default: 1)
- `--save-interval`: Save frequency for intermediate steps (default: steps/10)

**Hardware Configuration:**
- `--gpu`: GPU device ID (default: 0)

#### Usage Examples

**Process different cases:**
```bash
python visualize.py --input-case BDMAP_A0001000
python visualize.py --input-case BDMAP_A0001001
python visualize.py --input-case BDMAP_A0001002
```

**High-quality growth process:**
```bash
python visualize.py --input-case BDMAP_A0001000 --steps 50 --tumor-points 2 --save-interval 5
```

**Custom dataset location:**
```bash
python visualize.py \
    --input-case BDMAP_A0001000 \
    --dataset-root /path/to/your/AbdomenAtlasF \
    --output-root /path/to/output
```

#### Output Structure
```
output/growth_process/{case_id}/
├── images/
│   ├── tumor_step_000.nii.gz
│   ├── tumor_step_005.nii.gz
│   ├── ...
│   └── final_tumor.nii.gz
└── labels/
    ├── tumor_step_000.nii.gz
    ├── tumor_step_005.nii.gz
    ├── ...
    └── final_label_tumor.nii.gz
```

### Step 3: Create 3D Mesh Animations

Generate 3D mesh-based visualizations from tumor growth steps.

```bash
# Process default case
cd visualization
python 3Dgif.py

# Process specific case
python 3Dgif.py --input-case BDMAP_A0001001

# Custom paths
python 3Dgif.py \
    --input-case BDMAP_A0001000 \
    --growth-process-root ../../output/growth_process \
    --output-root ../../output
```

**Interactive Features:**
- Label selection: Choose which anatomical structures to display
- View angle configuration: Set horizontal and vertical viewing angles
- Animation direction: Forward (growth) or reverse (shrinkage)
- Multiple views: Generate different viewing angles

**Command Line Parameters:**
- `--input-case`: Case ID to visualize (default: BDMAP_A0001000)
- `--growth-process-root`: Path to growth process data (default: ../../output/growth_process)
- `--output-root`: Root output directory (default: ../../output)

**Output:**
```
output/3Dgif/{case_id}/
├── {case_id}_tumor_growth_front_45deg.gif
├── {case_id}_tumor_growth_side_view.gif
└── {case_id}_tumor_shrink_top_view.gif
```

### Step 4: Create 2D Slice Animations

Generate 2D slice-based animations showing tumor growth in different anatomical planes.

```bash
# Process default case
cd visualization
python tumor_gross2D_gif.py

# Process specific case
python tumor_gross2D_gif.py --input-case BDMAP_A0001001

# Custom duration
python tumor_gross2D_gif.py --input-case BDMAP_A0001000 --duration 300
```

**Command Line Parameters:**
- `--input-case`: Case ID to visualize (default: BDMAP_A0001000)
- `--growth-process-root`: Path to growth process data (default: ../../output/growth_process)
- `--output-root`: Root output directory (default: ../../output)
- `--duration`: Frame duration in milliseconds (default: 500)

**Features:**
- Automatic detection of largest tumor slice
- Side-by-side comparison (original vs. annotated)
- Three anatomical planes: sagittal, coronal, axial
- Contour highlighting for pancreas and tumor

**Output:**
```
output/gifs/{case_id}/
├── {case_id}_tumor_growth_largest_sagittal.gif
├── {case_id}_tumor_growth_largest_coronal.gif
└── {case_id}_tumor_growth_largest_axial.gif
```

## Complete Example Workflow

```bash
# Step 1: Merge segmentations (run once per case)
python merge_segmentations.py

# Step 2: Generate tumor growth process
python visualize.py --input-case BDMAP_A0001000 --steps 30

# Step 3: Create 3D animations
cd visualization
python 3Dgif.py --input-case BDMAP_A0001000

# Step 4: Create 2D animations  
python tumor_gross2D_gif.py --input-case BDMAP_A0001000

# Process another case
cd ..
python visualize.py --input-case BDMAP_A0001001
cd visualization
python 3Dgif.py --input-case BDMAP_A0001001
python tumor_gross2D_gif.py --input-case BDMAP_A0001001
```

## Visualization Customization

### 3D Visualization Settings

**Anatomical Structure Colors:**
- PDAC lesion: Red (100% opacity)
- Pancreatic parenchyma: Green (20% opacity)
- Pancreatic duct: Yellow (90% opacity)
- Veins: Blue (80% opacity)
- Arteries: Pink (80% opacity)
- Bile duct: Cyan (80% opacity)
- Pancreatic cyst: Light Purple (70% opacity)
- Pancreatic NET: Orange (90% opacity)
- Postcava: Dark Purple (80% opacity)

**Recommended View Combinations:**
- **Anterior view**: Horizontal: 0°, Vertical: 0°
- **Right lateral**: Horizontal: 90°, Vertical: 0°
- **Superior view**: Horizontal: 0°, Vertical: 90°
- **Oblique view**: Horizontal: 45°, Vertical: 30°

## Advanced Usage

### Batch Processing Multiple Cases
```bash
# Process multiple cases sequentially
for case in BDMAP_A0001000 BDMAP_A0001001 BDMAP_A0001002; do
    echo "Processing $case..."
    python visualize.py --input-case $case
    cd visualization
    python 3Dgif.py --input-case $case
    python tumor_gross2D_gif.py --input-case $case
    cd ..
done
```

### Custom Texture Generation
Modify texture parameters in `utils_visual.py`:

```python
# Adjust these parameters for different texture patterns
sigma_as = [3, 6, 9, 12, 15]  # Texture coarseness
sigma_bs = [4, 7]             # Texture smoothness
```

### Custom Growth Parameters
Modify growth parameters in `visualize.py`:

```python
# Adjust these for different growth characteristics
kernel_size = (3, 3, 3)      # Growth kernel size
threshold = 10               # Growth threshold
steps = 30                   # Growth steps
```

