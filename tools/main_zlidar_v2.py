import os, numpy as np, argparse, json, sys, yaml, multiprocessing, shutil
# sys.path.append("/home/liujianwei/project/share/gitlab/3dtracking-simpletrack/")
from matplotlib.lines import segment_hits
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import ZlidarLoaderV2
import pickle


parser = argparse.ArgumentParser()
# paths
parser.add_argument('--result_folder', type=str, default='../mot_results/')
parser.add_argument('--data_dets', type=str, default='')
# running configurations
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
args = parser.parse_args()

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
    # dets = [d for d in dets if d.s >= 0.1]
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


def sequence_mot(configs, data_loader: ZlidarLoaderV2, visualize=False, img_folder=''):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    IDs, bboxes, states, types = list(), list(), list(), list()
    det_update = []
    # det_num = len(det_data)
    
    for frame_index in range(data_loader.cur_frame, frame_num):
        print('TYPE {:} Frame {:} / {:}'.format(data_loader.type_token, frame_index + 1, frame_num))
        det_data = data_loader.data_segment[frame_index]['annotations_fusion']
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

        result_pred_bboxes = pred_content_filter(result_pred_bboxes, result_pred_states)
        result_pred_ids = pred_content_filter(result_pred_ids, result_pred_states)

        # visualization
        if visualize:
            frame_visualization(result_pred_bboxes, result_pred_ids, frame_data.pc, dets=frame_data.dets, name='{:}'.format(frame_index), img_folder=img_folder, vc=frame_data.vc)

        # update det data by adding tracking id
        for det in det_data:
            for res_index, res in enumerate(result_pred_bboxes):
                if det['id'] ==  res.index:
                    det['track_id'] = result_pred_ids[res_index]
                    break
        
        # wrap for output
        IDs.append(result_pred_ids)
        result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)
        det_update.append(det_data)

    return IDs, bboxes, states, types, det_update


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

    data_loader = ZlidarLoaderV2(configs, [type_token], data_segment, start_frame)
    
    # real mot happens here
    ids, bboxes, states, types, det_update = sequence_mot(configs, data_loader, args.visualize, img_folder)

    return ids, bboxes, states, types, det_update


if __name__ == '__main__':

    obj_types = ['vehicle', 'pedestrian', 'cyclist']

    # config_paths = ['../configs/zlidar_configs/vc_kf_giou_pc.yaml',
    #                 '../configs/zlidar_configs/pd_kf_giou_pc.yaml',
    #                 '../configs/zlidar_configs/cc_kf_giou_pc.yaml']

    config_paths = ['../configs/zlidar_configs/vc_kf_giou_nopc.yaml',
                '../configs/zlidar_configs/pd_kf_giou_nopc.yaml',
                '../configs/zlidar_configs/cc_kf_giou_nopc.yaml']

    # summary_folder = os.path.join(summary_folder, args.obj_type)
    # os.makedirs(summary_folder, exist_ok=True)

    with open(args.data_dets, 'rb') as f:
        dets = pickle.load(f)

    segment_names = sorted(list(dets.keys()))

    for index, segment_name in enumerate(segment_names):
        data_segment = dets[segment_name]
        frame_num = len(data_segment)
        track_results = dict()
        bboxes = [list() for _ in range(frame_num)]
        types = [list() for _ in range(frame_num)]
        ids = [list() for _ in range(frame_num)]
        dets_update = [list() for _ in range(frame_num)]
        print('START SEQ: ' + segment_name)
        for obj_type, config_path in zip(obj_types, config_paths):
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, index + 1, len(segment_names)))
            ids_cur, bboxes_cur, states_cur, types_cur, det_cur = main(obj_type, config_path, data_segment, segment_name, args.result_folder, args.start_frame)
            for frame_index in range(frame_num):
                for id, bbox, type, det in zip(ids_cur[frame_index], bboxes_cur[frame_index], types_cur[frame_index], det_cur[frame_index]):
                    bboxes[frame_index].append(list(bbox))
                    types[frame_index].append(type)
                    ids[frame_index].append(id)
                    dets_update[frame_index].append(det)

        for frame_index in range(frame_num):
            bboxes[frame_index] = np.array(bboxes[frame_index])

        # if index >= 1:
        #     break

        # for frame_index in range(frame_num):
        #     dets[segment_name][frame_index]['annotations_track'] = dets_update[frame_index]

        # track_results['bboxes'] = bboxes
        # track_results['types'] = types
        # track_results['ids'] = ids

    # with open(os.path.join(args.result_folder, '20221213105310700_to_20221213110010200.pkl'), 'wb') as f:
    #     pickle.dump(dets, f)

    with open(args.result_folder, 'wb') as f:
        pickle.dump(dets, f)

    

