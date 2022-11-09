det_name=$1
waymo_data_dir=$2
waymo_result_dir=$3
gt_dets_dir=$4
proc_num=$5

# for vehicle
python tools/main_waymo.py \
    --name SimpleTrack \
    --det_name ${det_name} \
    --obj_type vehicle \
    --config_path configs/waymo_configs/vc_kf_giou.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --gt_folder ${gt_dets_dir} \
    --process ${proc_num}

# for pedestrian
python tools/main_waymo.py \
    --name SimpleTrack \
    --det_name ${det_name} \
    --obj_type pedestrian \
    --config_path configs/waymo_configs/pd_kf_giou.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --gt_folder ${gt_dets_dir} \
    --process ${proc_num}

# for cyclist
python tools/main_waymo.py \
    --name SimpleTrack \
    --det_name ${det_name} \
    --obj_type cyclist \
    --config_path configs/waymo_configs/vc_kf_giou.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --gt_folder ${gt_dets_dir} \
    --process ${proc_num}