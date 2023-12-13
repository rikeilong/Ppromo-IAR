# Pose-promote: Progressive Visual Perception for Indoor Action Recognition

This repo is the official repository of "Pose-promote: Progressive Visual Perception for Indoor Action Recognition"

## Dependencies
* Pytorch≥1.10 with CUDA≥11.3
* numpy
* opencv-python
* pickle
* glob

## Dataset
Please prepare your dataset as the following structure (e.g., Toyata Smarthome):
```
-frames
    -Cook.Cleandishes_p02_r00_v02_c03
        - 1.jpeg
        - ...
        - L.jpeg
    -Cook.Cleandishes_p02_r00_v14_c03
        - 1.jpeg
        - ...
        - L.jpeg
    - ...

    -Walk_p13_r04_v05_c04
        - 1.jpeg
        - ...
        - L.jpeg

-pose
    -xsub
        - train_data_bone.npy
        - train_data_joint.npy
        - train_label.pkl
        - val_data_bone.npy
        - val_data_joint.npy
        - val_label.pkl
    -xview1
        - train_data_bone.npy
        - train_data_joint.npy
        - train_label.pkl
        - val_data_bone.npy
        - val_data_joint.npy
        - val_label.pkl
    -xview2
        - train_data_bone.npy
        - train_data_joint.npy
        - train_label.pkl
        - val_data_bone.npy
        - val_data_joint.npy
        - val_label.pkl

-pose_train
    -config
        -smarthome-cs
            - train_jonit.yaml
            - train_bone.yaml
            - test_jonit.yaml
            - test_bone.yaml
        - ...

-config
    -smarthome-cs
        - train.yaml
        - test.yaml
    - ...
```

## Visual Feature Extraction
```
-extract_visual_feat_14x14
    -extract_14x14_feat.py
```
* **Extraction**
```
cd extract_visual_feat_14x14
python extract_14x14_feat.py --output_dir /mnt/sda/smarthome_res18_14x14 --video_path /home/qilang/PythonProjects/ICME/frames/ --model resnet18
```
## Get Started (Take Toyota Smarthome Dataset as example)
First, we need to train the pose encoder individually. For more information, please go to [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
* **Training**
```bash
cd Ppromo-IAR
python /pose_train/main.py --config /pose_train/config/smarthome-cs/train_jonit.yaml
python run.py --config /config/smarthome-cs/train.yaml --save /weights/
```
* **Testing**
```bash
cd Ppromo-IAR
python test.py --config /config/smarthome-cs/test.yaml --model /weights/xxx.pt
```

# Note

* 2024.3.15 - We removed the segment removal function and adaptive late fusion strategy proposed in this paper.


     
# Citation
Please cite the following paper if you use this repository in your reseach.

    
# Contact
For any questions, feel free to contact: `lll5698@foxmail.com`
