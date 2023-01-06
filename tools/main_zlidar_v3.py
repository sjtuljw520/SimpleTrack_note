import os, numpy as np, argparse, json, sys, yaml, multiprocessing, shutil
from tracemalloc import start
# sys.path.append("/home/liujianwei/project/share/gitlab/3dtracking-simpletrack/")
from matplotlib.lines import segment_hits
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import ZlidarLoaderV3
import pickle


parser = argparse.ArgumentParser()
# paths
parser.add_argument('--result_folder', type=str, default='../mot_results/')
parser.add_argument('--data_dets', type=str, default='')
# running configurations
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
args = parser.parse_args()

tracklet_keep_thres = 0.6
min_tracklet_len = 6

label2name = {1:'Vehicle', 2:'Pedestrian', 3:'Cyclist'}

def decode_state(state_string):
    tokens = state_string.split('_')
    if tokens[0] == 'birth':
        return 1
    if tokens[0] == 'alive' and int(tokens[1]) == 1:
        return 1
    if tokens[0] == 'alive' and int(tokens[1]) == 3:
        return 1
    if tokens[0] == 'alive' and int(tokens[1]) == 0:
        return 0

def pred_content_filter(pred_contents, pred_states):
    result_contents = list()
    for content, state in zip(pred_contents, pred_states):
        if Validity.valid(state):
            result_contents.append(content)
    return result_contents


def frame_visualization(bboxes, ids, pc=None, dets=None, name='', img_folder='', vc=None):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12), vc=vc)
    if pc is not None:
        visualizer.handler_pc(pc)
    # for _, bbox in enumerate(gt_bboxes):
    #     visualizer.handler_box_gt(bbox, message='', color='black')
    dets = [d for d in dets if d.s >= 0.1]
    # for det in dets:
    #     visualizer.handler_box_gt(det, message='%.2f' % det.s, color='black', linestyle='dashed')
    for _, (bbox, id) in enumerate(zip(bboxes, ids)):
        # if Validity.valid(state):
        visualizer.handler_box(bbox, message=str(id), color_id=id)
        # else:
            # visualizer.handler_box(bbox, message=str(id), color='light_blue')
            # visualizer.handler_box(bbox, message=str(id), color_id=id)
    #visualizer.show()
    os.makedirs(img_folder, exist_ok=True)
    visualizer.save(os.path.join(img_folder,'{:}.png'.format(name)))
    visualizer.close()


def sequence_mot(configs, data_loader: ZlidarLoaderV3, visualize=False, img_folder=''):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    track_ids, bboxes, states, types = list(), list(), list(), list()
    # det_num = len(det_data)
    id_to_state_dict = dict()

    start_frame = data_loader.cur_frame
    
    for frame_index in range(data_loader.cur_frame, frame_num):
        print('TYPE {:} Frame {:} / {:}'.format(data_loader.type_token, frame_index + 1, frame_num))
        det_data = data_loader.data_segment[frame_index]['annotations']
        # input data
        frame_data = next(data_loader)
        frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'], 
            det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'], vc=frame_data['vc'])

        # mot
        results = tracker.frame_mot(frame_data)
        result_pred_bboxes = [trk[0] for trk in results]
        result_pred_ids = [trk[1] for trk in results]
        result_pred_states = [trk[2] for trk in results]
        result_types = [trk[3] for trk in results]

        # result_pred_bboxes = pred_content_filter(result_pred_bboxes, result_pred_states)
        # result_pred_ids = pred_content_filter(result_pred_ids, result_pred_states)

        # visualization
        # if visualize:
        #     frame_visualization(result_pred_bboxes, result_pred_ids, frame_data.pc, dets=frame_data.dets, name='{:}'.format(frame_index), img_folder=img_folder, vc=frame_data.vc)

        # update det data by adding tracking id
        # for det in det_data:
        #     for res_index, res in enumerate(result_pred_bboxes):
        #         if det['id'] ==  res.index:
        #             det['track_id'] = result_pred_ids[res_index]
        #             break
        
        # wrap for output
        track_ids.append(result_pred_ids)
        #result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)

        for index, id in enumerate(result_pred_ids):
            if id in id_to_state_dict:
                id_to_state_dict[id].append(decode_state(result_pred_states[index]))
            else:
                id_to_state_dict[id] = list()
                id_to_state_dict[id].append(decode_state(result_pred_states[index]))

    ids_keep = []

    for id, id_state in id_to_state_dict.items():
        if len(id_state) >= min_tracklet_len and sum(id_state) / len(id_state) >= tracklet_keep_thres:
            ids_keep.append(id)

    track_ids_keep, bboxes_keep, states_keep, types_keep =  list(), list(), list(), list()
    for frame_ids, frame_bboxes, frame_states, frame_types in zip(track_ids, bboxes, states, types):
        keep_index = [i for i, id in enumerate(frame_ids) if id in ids_keep]
        track_ids_keep.append([frame_ids[i] for i in keep_index])
        bboxes_keep.append([frame_bboxes[i] for i in keep_index])
        states_keep.append([frame_states[i] for i in keep_index])
        types_keep.append([frame_types[i] for i in keep_index])

    # track_ids_tune, bboxes_tune, states_tune, types_tune =  list(), list(), list(), list()
    process_num = len(track_ids_keep)
    ids_remove_dict = dict()
    for index in range(process_num-1,-1,-1):
        move_index = [i for i, state in enumerate(states_keep[index]) if decode_state(state) == 0]
        keep_index = [i for i, state in enumerate(states_keep[index]) if decode_state(state) > 0]
        for k_i in keep_index:
            id_keep = track_ids_keep[index][k_i]
            if id_keep in ids_remove_dict:
                ids_remove_dict[id_keep] = 'cannot_remove'
        real_move_index = []
        for m_i in move_index:
            id_move = track_ids_keep[index][m_i]
            if id_move not in ids_remove_dict:
                ids_remove_dict[id_move] = 'can_remove'
                real_move_index.append(m_i)
            else:
                if ids_remove_dict[id_move] == 'can_remove':
                    real_move_index.append(m_i)

        track_ids_keep[index] = np.delete(track_ids_keep[index],real_move_index).tolist()
        bboxes_keep[index] = np.delete(bboxes_keep[index],real_move_index).tolist()
        states_keep[index] = np.delete(states_keep[index],real_move_index).tolist()
        types_keep[index] = np.delete(types_keep[index],real_move_index).tolist()

    if visualize:
        print("Start to plot the figure...")
        data_loader.cur_frame = start_frame
        for frame_index in range(start_frame, frame_num):
            frame_data = next(data_loader)
            frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'], 
                det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'], vc=frame_data['vc'])
            frame_visualization(bboxes_keep[frame_index-start_frame], track_ids_keep[frame_index-start_frame], frame_data.pc, dets=frame_data.dets, name='{:}'.format(frame_index), img_folder=img_folder, vc=frame_data.vc)
        print("Finish the plotting.")

    # transform box from world to ego
    bboxes_ego = []
    ego_info = data_loader.ego_info
    for frame_index in range(start_frame, frame_num):
        inv_ego_motion = np.linalg.inv(ego_info[frame_index])
        bboxes_ego.append([BBox.bbox2world(inv_ego_motion, bbox) for bbox in bboxes_keep[frame_index-start_frame]])

    bboxes_array = []
    for frame_bboxes in bboxes_ego:
        bboxes_array.append([BBox.bbox2array(bbox) for bbox in frame_bboxes])

    return track_ids_keep, bboxes_array, states_keep, types_keep


def main(obj_type, config_path, data_segment, segment_name, result_folder, start_frame=0):
    # image folder if visualization
    img_folder = os.path.join(result_folder, 'imgs', segment_name, obj_type)
    # load model configs
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    
    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 3

    data_loader = ZlidarLoaderV3(configs, [type_token], data_segment, start_frame)
    
    # real mot happens here
    ids, bboxes, states, types = sequence_mot(configs, data_loader, args.visualize, img_folder)

    return ids, bboxes, states, types


if __name__ == '__main__':

    obj_types = ['vehicle', 'pedestrian', 'cyclist']

    # config_paths = ['./configs/zlidar_configs_v3/vc_kf_giou_pc.yaml',
    #                 './configs/zlidar_configs_v3/pd_kf_giou_pc.yaml',
    #                 './configs/zlidar_configs_v3/cc_kf_giou_pc.yaml']

    config_paths = ['./configs/zlidar_configs_v3/vc_kf_giou_nopc.yaml',
                './configs/zlidar_configs_v3/pd_kf_giou_nopc.yaml',
                './configs/zlidar_configs_v3/cc_kf_giou_nopc.yaml']

    # summary_folder = os.path.join(summary_folder, args.obj_type)
    # os.makedirs(summary_folder, exist_ok=True)

    with open(args.data_dets, 'rb') as f:
        dets = pickle.load(f)

    segment_names = sorted(list(dets.keys()))

    for seg_index, segment_name in enumerate(segment_names):
        data_segment = dets[segment_name]
        frame_num = len(data_segment)
        track_results = dict()

        print('START SEQ: ' + segment_name)
        for frame_index in range(frame_num):
            data_segment[frame_index]['annotations_track'] = []
        for obj_type, config_path in zip(obj_types, config_paths):
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, seg_index + 1, len(segment_names)))
            ids_cur, bboxes_cur, states_cur, types_cur = main(obj_type, config_path, data_segment, segment_name, args.result_folder, args.start_frame)
            for frame_index in range(args.start_frame, frame_num):
                index_cur = len(data_segment[frame_index]['annotations_track'])
                for id, bbox, type, state in zip(ids_cur[frame_index], bboxes_cur[frame_index], types_cur[frame_index], states_cur[frame_index]):
                    anno_dict = dict()
                    x, y, z, o, l, w, h = bbox[:7]
                    anno_dict['gt_boxes'] = [x,y,z,w,l,h,o,0,0]
                    anno_dict['gt_names'] = label2name[type]
                    anno_dict['gt_velocity'] = [0,0]
                    anno_dict['gt_acc'] = [0,0]
                    anno_dict['detection_score'] = bbox[7]
                    anno_dict['track_id'] = id
                    anno_dict['index'] = index_cur
                    anno_dict['state'] = decode_state(state)
                    data_segment[frame_index]['annotations_track'].append(anno_dict)
                    index_cur += 1

        # if seg_index >= 0:
        #     break


    # with open(os.path.join(args.result_folder, '20221213105310700_to_20221213110010200.pkl'), 'wb') as f:
    #     pickle.dump(dets, f)

    with open(args.result_folder, 'wb') as f:
        pickle.dump(dets, f)

    

