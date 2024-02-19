import os
import mot_3d.visualization as visualization
from mot_3d.data_protos import Validity
from mot_3d.frame_data import FrameData
from data_loader import ZlidarLoaderV4
import pickle
import time

label2name = {1:'Vehicle', 2:'Pedestrian', 3:'Cyclist'}

def pred_content_filter(pred_contents, pred_states):
    result_contents = list()
    for content, state in zip(pred_contents, pred_states):
        if Validity.valid(state):
            result_contents.append(content)
    return result_contents


def frame_visualization(pc=None, dets=None, name='', img_folder='', vc=None):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12), vc=vc)
    if pc is not None:
        visualizer.handler_pc(pc)
    for bbox in dets:
        # if Validity.valid(state):
        visualizer.handler_box(bbox, message=str(int(bbox.index)), color_id=int(bbox.index))
        # else:
            # visualizer.handler_box(bbox, message=str(id), color='light_blue')
            # visualizer.handler_box(bbox, message=str(id), color_id=id)
    #visualizer.show()
    os.makedirs(img_folder, exist_ok=True)
    visualizer.save(os.path.join(img_folder,'{:}.png'.format(name)))
    visualizer.close()


def main():

    obj_types = ['vehicle', 'pedestrian', 'cyclist']

    data_file = '/home/liujianwei/project/code/bev_anno/hat_scipts_v2/mot_results_v6/20221128145440695_to_20221128154034600_track_merge.pkl'
    result_folder = '/home/liujianwei/project/code/bev_anno/hat_scipts_v2/mot_results_v6/'

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("Loading data...")
    with open(data_file, 'rb') as f:
        dets = pickle.load(f)

    segment_names = sorted(list(dets.keys()))

    for seg_index, segment_name in enumerate(segment_names):
        data_segment = dets[segment_name]
        frame_num = len(data_segment)

        for obj_type in obj_types:
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, seg_index + 1, len(segment_names)))
            # image folder if visualization
            img_folder = os.path.join(result_folder, 'imgs_demo3', segment_name, obj_type)
            # load model configs  
            if obj_type == 'vehicle':
                type_token = 1
            elif obj_type == 'pedestrian':
                type_token = 2
            elif obj_type == 'cyclist':
                type_token = 3

            data_loader = ZlidarLoaderV4([type_token], data_segment, start_frame=0)

            for frame_index in range(data_loader.cur_frame, frame_num):
                # input data
                frame_data = next(data_loader)
                frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'], 
                    det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'], vc=frame_data['vc'])

                frame_visualization(frame_data.pc, dets=frame_data.dets, name='{:}'.format(frame_index), img_folder=img_folder, vc=frame_data.vc)

        if seg_index >= 0:
            break


if __name__ == '__main__':

    main()