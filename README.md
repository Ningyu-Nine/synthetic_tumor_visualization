# Synthetic Tumor Visualization

A tool for visualizing synthetic tumor growth process in medical imaging, specifically designed for pancreatic tumor visualization.

## Features

- Tumor growth simulation with adjustable parameters
- 3D visualization of the tumor growth process
- Support for multiple anatomical structures visualization
- Customizable viewing angles and animation settings
- Both forward (growth) and reverse (shrinkage) animation generation
- Transparency support for better structure visualization

## Directory Structure

```
tumor_visualization/
├── Tumor_Synthesis/
│   ├── visualization/
│   │   └── 3Dgif.py         # 3D visualization and GIF generation
│   ├── utils_visual.py      # Utility functions for tumor generation
│   └── visualize.py         # Main script for tumor synthesis
├── image/                   # Input CT images
├── label/                   # Input segmentation labels
└── output/
    ├── growth_process/
    │   ├── images/         # Generated tumor images
    │   └── labels/         # Generated tumor labels
    └── 3Dgif/             # Output GIF animations
```

## Prerequisites

- Python 3.7+
- PyTorch
- VTK
- SimpleITK
- NumPy
- OpenCV
- PIL

## Installation

```bash
pip install torch vtk SimpleITK numpy opencv-python pillow
```

## Usage

1. **Generate Synthetic Tumor Growth Process**:
```bash
python visualize.py
```
- Enter the total number of growth steps (recommended: 5-70)
- The program will generate intermediate results at regular intervals

2. **Create 3D Visualization GIF**:
```bash
python 3Dgif.py
```
- Select structures to visualize (tumor, vessels, pancreas, etc.)
- Choose viewing angles
- Select animation direction (growth or shrinkage)

## Using with AbdomenAtlasF Dataset

This tool also supports visualization of the AbdomenAtlasF dataset which has separate segmentation files for different organs.

### Quick Guide for AbdomenAtlasF:

1. **Merge segmentation files**:
```bash
python merge_segmentations.py
```

2. **Generate tumor growth process** (if needed):
```bash
python visualize.py
```

3. **Create 3D visualization**:
```bash
python 3Dgif.py
```

### Label Mapping
When using AbdomenAtlasF dataset, the following label mapping is used:
- 1: PDAC lesion (Red)
- 2: Veins (Blue)
- 3: Arteries (Pink)
- 4: Pancreas (Green)
- 5: Pancreatic duct (Yellow)
- 6: Bile duct (Cyan)
- 7: Pancreatic cyst (Light Purple)
- 8: Pancreatic NET (Orange)
- 9: Postcava (Dark Purple)
- 10: Superior Mesenteric Artery (Gold)

## Visualization Options

### Anatomical Structures
- PDAC lesion (Red)
- Veins (Blue)
- Arteries (Pink)
- Pancreatic parenchyma (Green)
- Pancreatic duct (Yellow)
- Bile duct (Cyan)
- Pancreatic cyst (Light Purple)
- Pancreatic NET (Orange)
- Postcava (Dark Purple)
- Superior Mesenteric Artery (Gold)

### Transparency Settings
- Tumor (PDAC): 100% opacity
- Pancreatic parenchyma: 20% opacity
- Pancreatic duct: 90% opacity
- Vessels and bile duct: 80% opacity
- Pancreatic cyst: 70% opacity
- Pancreatic NET: 90% opacity
- Postcava: 80% opacity
- Superior Mesenteric Artery: 100% opacity

## Output Files

- Intermediate tumor growth states (NIfTI format)
- 3D visualization animations (GIF format)
- Both image and label files for each growth step

## Contributing

This is a research tool for synthetic tumor visualization. 
