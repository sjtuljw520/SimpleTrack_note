import os, numpy as np, argparse, json, sys, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import ZlidarLoader
import pickle


parser = argparse.ArgumentParser()
# paths
parser.add_argument('--result_folder', type=str, default='../mot_results/')
parser.add_argument('--data_folder', type=str, default='../datasets/waymo/mot/')
# running configurations
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
args = parser.parse_args()


def frame_visualization(bboxes, ids, states, pc=None, dets=None, name='', img_folder='', vc=None):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12), vc=vc)
    if pc is not None:
        visualizer.handler_pc(pc)
    # for _, bbox in enumerate(gt_bboxes):
    #     visualizer.handler_box_gt(bbox, message='', color='black')
    # dets = [d for d in dets if d.s >= 0.1]
    # for det in dets:
    #     visualizer.handler_box_gt(det, message='%.2f' % det.s, color='black', linestyle='dashed')
    for _, (bbox, id, state) in enumerate(zip(bboxes, ids, states)):
        if Validity.valid(state):
            visualizer.handler_box(bbox, message=str(id), color_id=id)
        else:
            # visualizer.handler_box(bbox, message=str(id), color='light_blue')
            visualizer.handler_box(bbox, message=str(id), color_id=id)
    #visualizer.show()
    os.makedirs(img_folder, exist_ok=True)
    visualizer.save(os.path.join(img_folder,'{:}.png'.format(name)))
    visualizer.close()


def sequence_mot(configs, data_loader: ZlidarLoader, sequence_id, visualize=False, img_folder=''):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    IDs, bboxes, states, types = list(), list(), list(), list()
    for frame_index in range(data_loader.cur_frame, frame_num):
        print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(data_loader.type_token, sequence_id + 1, frame_index + 1, frame_num))
        
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

        # visualization
        if visualize:
            frame_visualization(result_pred_bboxes, result_pred_ids, result_pred_states, frame_data.pc, dets=frame_data.dets, name='{:}'.format(frame_index), img_folder=img_folder, vc=frame_data.vc)
        
        # wrap for output
        IDs.append(result_pred_ids)
        result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)
    return IDs, bboxes, states, types


def main(obj_type, config_path, data_folder, file_name, result_folder, start_frame=0):
    # image folder if visualization
    img_folder = os.path.join(result_folder, 'imgs', obj_type)
    # load model configs
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    
    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 3

    segment_name = file_name.split('.')[0]
    data_loader = ZlidarLoader(configs, [type_token], segment_name, data_folder, start_frame)
    
    # real mot happens here
    ids, bboxes, states, types = sequence_mot(configs, data_loader, file_index, args.visualize, img_folder)

    return ids, bboxes, states, types


if __name__ == '__main__':

    obj_types = ['vehicle', 'pedestrian', 'cyclist']

    config_paths = ['../configs/waymo_configs/vc_kf_giou.yaml',
                    '../configs/waymo_configs/pd_kf_giou.yaml',
                    '../configs/waymo_configs/cc_kf_giou.yaml']

    # config_paths = ['../configs/waymo_configs/vc_kf_giou_nopc.yaml',
    #             '../configs/waymo_configs/pd_kf_giou_nopc.yaml',
    #             '../configs/waymo_configs/cc_kf_giou_nopc.yaml']

    # summary_folder = os.path.join(summary_folder, args.obj_type)
    # os.makedirs(summary_folder, exist_ok=True)

    file_names = sorted(os.listdir(os.path.join(args.data_folder, 'ego_info'))) # segment names

    for file_index, file_name in enumerate(file_names[:]):
        segment_name = file_name.split('.')[0]

        result_folder = os.path.join(args.result_folder, segment_name)
        os.makedirs(result_folder, exist_ok=True)
        summary_folder = os.path.join(result_folder, 'summary')
        os.makedirs(summary_folder, exist_ok=True)

        with open(os.path.join(args.data_folder, 'detection', '{:}.pkl'.format(segment_name)), 'rb') as f:
            dets = pickle.load(f)
        pcd_names = dets['names']

        frame_num = len(pcd_names)
        track_results = dict()
        bboxes = [list() for _ in range(frame_num)]
        types = [list() for _ in range(frame_num)]
        ids = [list() for _ in range(frame_num)]

        for obj_type, config_path in zip(obj_types, config_paths):
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1, len(file_names)))
            ids_cur, bboxes_cur, states_cur, types_cur = main(obj_type, config_path, args.data_folder, file_name, result_folder, args.start_frame)
            for frame_index in range(frame_num):
                for id, bbox, type in zip(ids_cur[frame_index], bboxes_cur[frame_index], types_cur[frame_index]):
                    bboxes[frame_index].append(list(bbox))
                    types[frame_index].append(type)
                    ids[frame_index].append(id)

        for frame_index in range(frame_num):
            bboxes[frame_index] = np.array(bboxes[frame_index])

        track_results['bboxes'] = bboxes
        track_results['types'] = types
        track_results['ids'] = ids
        track_results['names'] = pcd_names

        save_file = os.path.join(summary_folder,'track_results.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(track_results, f)