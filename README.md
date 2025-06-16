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
3. 3Dgif.py → Create 3D mesh animations
4. tumor_gross2D_gif.py → Create 2D slice animations
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
│   ├── 3Dgif/                   # 3D animations
│   └── gifs/                    # 2D animations
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
python visualize.py --input-dir /path/to/merged/data --steps 30 --tumor-points 2
```

#### Command Line Parameters

##### Required Parameters
- `--input-dir`: Directory containing merged image/ and label/ subdirectories

##### Optional Parameters

**Tumor Growth Parameters:**
- `--steps`: Total growth steps (default: 30)
  - Range: 5-100 recommended
  - Higher values = smoother transitions
  - Example: `--steps 50`

- `--tumor-points`: Initial tumor seed points (default: 1)
  - Range: 1-10
  - Multiple points create multi-focal tumors
  - Example: `--tumor-points 3`

**Hardware Configuration:**
- `--gpu`: GPU device ID (default: 0)
  - Use -1 for CPU-only mode
  - Example: `--gpu 1`

**Output Control:**
- `--save-interval`: Save frequency for intermediate steps (default: 0)
  - When 0: automatically sets to steps/10
  - Controls storage vs. detail balance
  - Example: `--save-interval 5`

#### Usage Examples

**Basic Usage:**
```bash
python visualize.py --input-dir /data/BDMAP_A0001000
```

**High-Quality Growth Process:**
```bash
python visualize.py \
    --input-dir /data/BDMAP_A0001000 \
    --steps 50 \
    --tumor-points 2 \
    --save-interval 5
```



#### Output Structure
```
output/growth_process/ct.nii/
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
python visualization/3Dgif.py
```

**Interactive Features:**
- Label selection: Choose which anatomical structures to display
- View angle configuration: Set horizontal and vertical viewing angles
- Animation direction: Forward (growth) or reverse (shrinkage)
- Multiple views: Generate different viewing angles

**Viewing Angle Guide:**
- **Horizontal angle (Azimuth)**: -180° to 180°
  - 0° = front view
  - 90° = right side view
  - 180° = back view
- **Vertical angle (Elevation)**: -90° to 90°
  - 0° = horizontal view
  - 90° = top view
  - -90° = bottom view

**Example Sessions:**
```
Select labels to display: 1,2,4,5
Select animation order: 1 (Forward - growth)
Horizontal angle: 45
Vertical angle: 30
View name: front_45deg
```

**Output:**
```
output/3Dgif/
├── tumor_growth_front_45deg.gif
├── tumor_growth_side_view.gif
└── tumor_shrink_top_view.gif
```

### Step 4: Create 2D Slice Animations

Generate 2D slice-based animations showing tumor growth in different anatomical planes.

```bash
python visualization/tumor_gross2D_gif.py
```

**Features:**
- Automatic detection of largest tumor slice
- Side-by-side comparison (original vs. annotated)
- Three anatomical planes: sagittal, coronal, axial
- Contour highlighting for pancreas and tumor

**Output:**
```
output/gifs/
├── tumor_growth_largest_sagittal.gif
├── tumor_growth_largest_coronal.gif
└── tumor_growth_largest_axial.gif
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
- **Anterior view**: `--angle-h 0 --angle-v 0`
- **Right lateral**: `--angle-h 90 --angle-v 0`
- **Superior view**: `--angle-h 0 --angle-v 90`
- **Oblique view**: `--angle-h 45 --angle-v 30`




## Advanced Usage

### Batch Processing Multiple Cases
Modify paths in `visualize.py` to process multiple AbdomenAtlasF cases:

```python
# In visualize.py, modify these paths:
cases = ['BDMAP_A0001000', 'BDMAP_A0001001', ...]
for case in cases:
    process_case(case)
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
