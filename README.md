# RoofVE
Rule-based roof vertex detection from ALS point clouds.  
Code repository of [Detecting vertices of building roofs from ALS point cloud data](https://www.tandfonline.com/doi/full/10.1080/17538947.2023.2283486)

## Repository structure
```text
.
|- config                   # config files.
    |- sampled_roof_50.txt  # the .txt file which includes the roof list will be considered.
|- Data
    |- las-singleroof-50    # point clouds of 50 sampled roofs
    |- las-singleroof-GT    # ground truth of 3D roof structures.
|- res                      # result output folder
|- utils                    # utilities to support the main function.
|- main_RoofV_det.py        # main script to detect roof vertices.
|- eval_NoOver.py           # evaluation script
|- README.md
|- requirement.txt
```

## Usage
### Run and eval
**For detecting roof vertices:**  
use through IDE by running `main_RoofV_det.py`  
**For evaluation:**  
use through IDE by running `eval_NoOver.py`  

### Requirements
The following requirements are necessary for this repository
- scipy
- sklearn
- laspy
- open3d
- shapely
- alphashape
- opencv-python
- triangle

Other potential requirements can be found in [requirements.txt](requirements.txt) or according to the compilation errors.

## Data
For speeding up the download of source codes, the data is provided separately via [Google drive](https://drive.google.com/drive/folders/1P-ijrKTCKk-jWDjs4WjqQJbs0n6TSd_x?usp=sharing)

The experimental result on the provided custom dataset is also available through the same link.