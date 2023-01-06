# for vehicle
python tools/main_zlidar.py \
    --name segment1 \
    --obj_type vehicle \
    --config_path configs/waymo_configs/vc_kf_giou.yaml \
    --data_folder /mnt/data/zlidar/lidar-sample \
    --result_folder /mnt/data/zlidar/lidar-sample/mot_results \
    --process 1 \
    --visualize

# # for pedestrian
python tools/main_zlidar.py \
    --name segment1 \
    --obj_type pedestrian \
    --config_path configs/waymo_configs/pd_kf_giou.yaml \
    --data_folder /mnt/data/zlidar/lidar-sample \
    --result_folder /mnt/data/zlidar/lidar-sample/mot_results \
    --process 1 \
    --visualize

# # for cyclist
python tools/main_zlidar.py \
    --name segment1 \
    --obj_type cyclist \
    --config_path configs/waymo_configs/cc_kf_giou.yaml \
    --data_folder /mnt/data/zlidar/lidar-sample \
    --result_folder /mnt/data/zlidar/lidar-sample/mot_results \
    --process 1 \
    --visualize