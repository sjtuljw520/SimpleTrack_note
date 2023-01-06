det_name=$1
waymo_data_dir=$2
waymo_result_dir=$3
proc_num=$4

# for vehicle
python tools/main_waymo_test.py \
    --name st_mppnet \
    --det_name ${det_name} \
    --obj_type vehicle \
    --config_path configs/waymo_configs/vc_kf_giou_nopc.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --process ${proc_num}

# for pedestrian
python tools/main_waymo_test.py \
    --name st_mppnet \
    --det_name ${det_name} \
    --obj_type pedestrian \
    --config_path configs/waymo_configs/pd_kf_giou_nopc.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --process ${proc_num}

# for cyclist
python tools/main_waymo_test.py \
    --name st_mppnet \
    --det_name ${det_name} \
    --obj_type cyclist \
    --config_path configs/waymo_configs/cc_kf_giou_nopc.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --process ${proc_num}