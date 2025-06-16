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

**CPU-Only Mode:**
```bash
python visualize.py \
    --input-dir /data/BDMAP_A0001000 \
    --steps 20 \
    --gpu -1
```

**Multi-focal Tumor:**
```bash
python visualize.py \
    --input-dir /data/BDMAP_A0001000 \
    --steps 40 \
    --tumor-points 5 \
    --save-interval 3
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

## Parameter Recommendations

### For Different Hardware Configurations

**GPU Memory < 8GB:**
```bash
--steps 20 --tumor-points 1 --save-interval 4
```

**GPU Memory 8-16GB:**
```bash
--steps 30-50 --tumor-points 2-3 --save-interval 5
```

**GPU Memory > 16GB:**
```bash
--steps 50-100 --tumor-points 3-5 --save-interval 3
```

**CPU Only:**
```bash
--steps 10-20 --tumor-points 1 --gpu -1
```

### For Different Use Cases

**Quick Preview:**
```bash
--steps 10 --tumor-points 1 --save-interval 2
```

**Research Quality:**
```bash
--steps 50 --tumor-points 2 --save-interval 3
```

**Publication Quality:**
```bash
--steps 70-100 --tumor-points 3 --save-interval 2
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

## Common Issues and Solutions

### GPU Memory Issues
```
Error: CUDA out of memory
Solution: Reduce --steps or use --gpu -1 for CPU mode
```

### Slow Processing
```
Issue: Processing takes too long
Solution: Reduce --steps or increase --save-interval
```

### Missing Input Files
```
Error: FileNotFoundError
Solution: Run merge_segmentations.py first
```

### Empty Visualization
```
Issue: No structures visible in 3D view
Solution: Check label selection and ensure merged labels are correct
```

## Performance Optimization

### Memory Management
- Monitor GPU memory with `nvidia-smi`
- Use appropriate batch sizes based on available memory
- Clear CUDA cache between runs if needed

### Speed Optimization
- Use GPU acceleration when available
- Adjust `--save-interval` to balance storage and processing time
- Consider using fewer tumor points for faster processing

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

## Troubleshooting

### Environment Issues
1. **VTK Import Error**: Ensure VTK is installed with Python bindings
2. **CUDA Issues**: Check PyTorch CUDA compatibility
3. **Memory Issues**: Monitor system and GPU memory usage

### Data Issues
1. **File Format**: Ensure all files are in NIfTI format (.nii.gz)
2. **Image Alignment**: Verify that CT and segmentation files are aligned
3. **Label Values**: Check that merged labels use correct integer values

### Visualization Issues
1. **Empty Renders**: Verify label selection and data integrity
2. **Incorrect Colors**: Check label mapping in `get_label_properties()`
3. **Performance**: Adjust mesh quality and rendering parameters

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{synthetic_tumor_visualization,
  title={Synthetic Tumor Visualization Tool},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/synthetic-tumor-visualization}
}
```

## Contributing

This is a research tool for synthetic tumor visualization. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
