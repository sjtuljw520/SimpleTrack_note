# 3DTracking-simpletrack

Lidar 3D object tracking with simpletrack algorithm. 

## Installation
### Requirements
The codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04)
* Python 3.8.13
* PyTorch 1.8.1+cu111
* CUDA 11.1 or higher

### Install
* cd  current folder.
* pip install -r requirements.txt
* pip install -e ./
  
## Run the code

本代码依赖于检测模型的输出，需要先用PVRCNN++算法做3D目标检测。

### 数据预处理

* 将点云数据按片段做预处理，每一片段包含若干帧连续点云数据、每帧点云的车辆Pose以及每帧点云的时间戳信息。
* 将检测输出按片段做预处理，每一片段包含若干帧点云连续点云的3D检测结果。

### 推理示例

```bash
cd tools/
python main_zlidar.py --data_folder /mnt/data/zlidar/lidar-sample --result_folder /mnt/data/zlidar/lidar-sample/mot_results
```
  * 需要可视化时，用如下配置文件：
```python
    config_paths = ['../configs/waymo_configs/vc_kf_giou.yaml',
                    '../configs/waymo_configs/pd_kf_giou.yaml',
                    '../configs/waymo_configs/cc_kf_giou.yaml']
```
* 不做可视化时，用如下配置文件：
```python
    config_paths = ['../configs/waymo_configs/vc_kf_giou_nopc.yaml',
                '../configs/waymo_configs/pd_kf_giou_nopc.yaml',
                '../configs/waymo_configs/cc_kf_giou_nopc.yaml']
```
* 目录/mnt/data/zlidar/lidar-sample说明
```python
/mnt/data/zlidar/lidar-sample/
├── detection                # 点云目标检测结果保存目录
│   └── segment1.pkl         # segment1是段名，表示一段连续帧点云
│                            # pkl文件包含一个字典，有三个keys['bboxes', 'types', 'names']
│                            # ['bboxes'] 是一个列表，列表长度等于该段点云的帧数，
│                            # ['bboxes'] 列表的每个元素：N*8 [x,y,z,yaw,l,w,h,score]矩阵
│                             
├── ego_info                 # 车辆位姿信息
│   └── segment1.pkl         # pkl里的数据是列表，列表长度等于该段点云的帧数，每个元素是 4*4 矩阵
│
├── lidar_info               # lidar2ego的外参，读取点云数据时转到ego坐标系时用到
│   └── segment1.pkl
├── mot_results              # 目标跟踪结果文件夹
│   └── segment1             # 段名
│       ├── imgs             # 可视化时保存的图片目录
│       │   ├── cyclist
│       │   ├── pedestrian
│       │   └── vehicle
│       └── summary         # 结果数据目录
│           ├── track_results.pkl   # 相比detection目录下的pkl数据，增加了key['ids']表示对象id
│ 
├── pc                      # 点云文件目录，可视化时需要用到
│   └── segment1
│       ├── n000001_2022-12-16-15-28-45-000076_Pandar128.pcd
│       ├── n000001_2022-12-16-15-28-45-100081_Pandar128.pcd
│       ├── ...
└── ts_info                # 时间戳信息
    └── segment1.pkl       # pkl里的数据是列表，列表长度等于该段点云的帧数
```

### 相关数据卷

- clever/lidar-liu/lidar-sample.rar 存放着上述示例数据

### main_zlidar_v2.py 说明

这个版本的跟踪算法，是先跟图像融合标注后，再跟踪。

dataloader 版本：./data_loader/zlidar_loader.py/class lidarLoaderV2.

### main_zlidar_v3.py 说明

这个版本的跟踪算法，是先点云3框做标注，再跟图像融合。

dataloader 版本：./data_loader/zlidar_loader.py/class lidarLoaderV3.

### 备注

- main_zlidar_v2.py和main_zlidar_v3.py需要跟融合标注程序（lidar_img_fusion_v2）配合使用，其输入ts_info、ego_info和detection都已打包在同一个文件，同时做完跟踪后，需要用lidar_img_fusion_v2的后处理代码做后处理。
- 运行示例见上一层目录的readme。

