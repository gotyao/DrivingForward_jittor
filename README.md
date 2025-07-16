# DrivingForward(Jittor)
This is a jittor implementation of [DrivingForward](https://github.com/fangzhou2000/DrivingForward). 

## Installation
```
git clone https://github.com/gotyao/DrivingForward_jittor
cd DrivingForward_jittor
conda create -n DFjittor python=3.9
conda activate DFjittor
pip install jittor
pip install -r requirements.txt
cd models/gaussian/gaussian-renderer/diff_gaussian_rasterizater
cmake .
make -j
cd ../../scene/simple-knn
cmake .
make -j
cd ../../../..
```

Note: The repository uses [Jittor_Perceptual-Similarity-Metric](https://github.com/ty625911724/Jittor_Perceptual-Similarity-Metric) for evaluation. Please download the pretrained model from the source repository and place them within the `DFjittor` folder.

## Datasets

### nuScenes 
* Download [nuScenes](https://www.nuscenes.org/nuscenes) official dataset
* Place the dataset in `input_data/nuscenes/`

Data should be as follows:
```
├── input_data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
```

## Running the Code

### Evaluation

Get the [pretrained models](https://drive.google.com/file/d/1zT7FzUihjax1S5v_iuWJqc70DON76U1v/view?usp=sharing), save them to the root directory of the project, and unzip them.

For SF mode, run the following:
```
python -W ignore eval.py --weight_path ./weights_SF --novel_view_mode SF
```

For MF mode, run the following:
```
python -W ignore eval.py --weight_path ./weights_MF --novel_view_mode MF
```

### Training

For SF mode, run the following:
```
python -W ignore train.py --novel_view_mode SF
```

For MF mode, run the following:
```
python -W ignore train.py --novel_view_mode MF
```

## Citation
```
@inproceedings{tian2025drivingforward,
    title={DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input}, 
    author={Qijian Tian and Xin Tan and Yuan Xie and Lizhuang Ma},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
}
```

## Acknowledgements
This implementation is based on the following project:   
[Dataset-Governance-Policy](https://github.com/TRI-ML/dgp)  
[PackNet-SfM](https://github.com/TRI-ML/packnet-sfm)  
[gaussian-splatting-jittor](https://github.com/otakuxiang/gaussian-splatting-jittor)  
[Jittor_Perceptual-Similarity-Metric](https://github.com/ty625911724/Jittor_Perceptual-Similarity-Metric)  
