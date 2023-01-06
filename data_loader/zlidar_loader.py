""" Example of data loader:
    The data loader has to be an iterator:
    Return a dict of frame data
    Users may create the logic of your own data loader
"""
import os, numpy as np, json
import mot_3d.utils as utils
from mot_3d.data_protos import BBox
from mot_3d.preprocessing import nms
import pickle
import open3d

Extrinsics = np.array([[1.06805951e-03,-9.99857909e-01,-1.68729842e-02,1.07098034e+00],
                        [9.99810288e-01,7.39604419e-04,1.94593057e-02,-4.65892236e-02],
                        [-1.94438655e-02,-1.68905455e-02,9.99667673e-01,1.95128934e+00],
                        [0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]])

def process_pcd_data_binary_compose(file_path, lidar2ego):
    pcd = open3d.t.io.read_point_cloud(filename=file_path)
    positions = pcd.point.positions.numpy()
    intensity = pcd.point.intensity.numpy()
    points = np.hstack((positions, intensity)).astype(np.float64)
    points_xyz = np.zeros(points.shape, dtype=float)
    if points.size > 0:
        points_xyz = np.concatenate((points[:,:3],np.ones((points.shape[0],1))),axis=1)
        points_xyz = points_xyz.dot(lidar2ego.T)
        points_xyz[:,3] = points[:,3] /255.0
        #points_xyz = remove_ego_points(points_xyz, x_radius=1.5, y_radius=1.0)
    else:
        print("file has no data: " + file_path)
    return points_xyz[:,:3]

class ZlidarLoader:
    def __init__(self, configs, type_token, segment_name, data_folder, start_frame):
        """ initialize with the path to data 
        Args:
            data_folder (str): root path to your data
        """
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.type_token = type_token

        self.nms = configs['data_loader']['nms']
        self.nms_thres = configs['data_loader']['nms_thres']

        with open(os.path.join(data_folder, 'ts_info', '{:}.pkl'.format(segment_name)), 'rb') as f:
            self.ts_info = pickle.load(f)
        with open(os.path.join(data_folder, 'ego_info', '{:}.pkl'.format(segment_name)), 'rb') as f:
            self.ego_info = pickle.load(f)
        # with open(os.path.join(data_folder, 'lidar_info', '{:}.pkl'.format(segment_name)), 'rb') as f:
        #     self.lidar_info = pickle.load(f)
        self.lidar_info = Extrinsics
        with open(os.path.join(data_folder, 'detection', '{:}.pkl'.format(segment_name)), 'rb') as f:
            self.dets = pickle.load(f)
        self.det_type_filter = True
        
        self.use_pc = configs['data_loader']['pc']
        if self.use_pc:
            self.pc_dir = os.path.join(data_folder, 'pc', segment_name)

        self.max_frame = len(self.dets['bboxes'])
        self.cur_frame = start_frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration

        result = dict()
        result['time_stamp'] = self.ts_info[self.cur_frame] * 1e-6
        result['ego'] = self.ego_info[self.cur_frame]

        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        selected_dets = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['dets'] = [BBox.bbox2world(result['ego'], BBox.array2bbox(b))
            for b in selected_dets]

        result['pc'] = None
        if self.use_pc:
            pc_file = os.path.join(self.pc_dir,self.dets['names'][self.cur_frame])
            pc = process_pcd_data_binary_compose(pc_file, self.lidar_info)
            # print(pc.shape)
            # print(pc[:3,:])
            result['pc'] = utils.pc2world(result['ego'], pc)
        
        result['aux_info'] = {'is_key_frame': True}
        if 'velos' in self.dets.keys():
            cur_frame_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [np.array(cur_frame_velos[i]) 
                for i in range(len(bboxes)) if inst_types[i] in self.type_token]
            result['aux_info']['velos'] = [utils.velo2world(result['ego'], v) 
                for v in result['aux_info']['velos']]
        else:
            result['aux_info']['velos'] = None
        
        if self.nms:
            result['dets'], result['det_types'], result['aux_info']['velos'] = \
                self.frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], self.nms_thres)
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]
        #print(np.ndarray(shape=(1,3), buffer=np.array([0,0,0]), dtype=float))
        result['vc'] = utils.pc2world(result['ego'], np.ndarray(shape=(1,3), buffer=np.array([0,0,0]), dtype=float))

        self.cur_frame += 1
        return result
    
    def __len__(self):
        return self.max_frame
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos
