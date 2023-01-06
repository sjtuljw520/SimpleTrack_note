import argparse
import glob
from pathlib import Path
import pickle
import os

import numpy as np
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

def trans2world(pc_data, ego2word):
    ones = np.ones((pc_data.shape[0], 1))
    points_expand = np.concatenate((pc_data,ones), axis=1)
    points_new = points_expand.dot(ego2word.T)
    return points_new[:,:3]


def vis_pc(pc_points):
    # 
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc_points)
    pcd_new = open3d.geometry.PointCloud.uniform_down_sample(pcd, 40)
    print(pcd_new)
    open3d.visualization.draw_geometries([pcd_new])

def draw_scenes(points, point_colors=None, draw_origin=True):
    vis = open3d.visualization.Visualizer()
    # vis = open3d.visualization.O3DVisualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    # pts = open3d.geometry.PointCloud.uniform_down_sample(pts, 40)
    print(pts)
    print(np.array(pts.points).shape)

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((np.array(pts.points).shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    vis.run()
    vis.destroy_window()


def splicing_lidar(lidar_path,lidar2ego_f,ego2world_f):

    lidar_path = Path(lidar_path)
    lidar2ego_f = open(lidar2ego_f, 'rb')
    ego2world_f = open(ego2world_f, 'rb')
    ext = '.pcd'

    data_file_list = glob.glob(str(lidar_path / f'*{ext}')) if lidar_path.is_dir() else [lidar_path]
    data_file_list.sort()

    lidar2ego = pickle.load(lidar2ego_f)
    ego2world = pickle.load(ego2world_f)

    lidar2ego = lidar2ego.values
    ego2world = ego2world.values

    lidar2ego_new = []
    ego2world_new = []

    for index in range(lidar2ego.shape[0]-1):
        lidar2ego_new.append(np.reshape(lidar2ego[index+1],(4,4)))

    for index in range(ego2world.shape[0]-1):
        ego2world_new.append(np.reshape(ego2world[index+1],(4,4)))

    assert len(data_file_list) == len(lidar2ego_new) == len(ego2world_new)

    pc_all_frames = []

    for index in range(24,len(data_file_list)):
        pc_ego = process_pcd_data_binary_compose(data_file_list[index],lidar2ego_new[index])
        pc_world = trans2world(pc_ego, ego2world_new[index])
        pc_all_frames.append(pc_world)

    pc_all_frames = np.concatenate(pc_all_frames, axis=0)

    return pc_all_frames

def trans_data(lidar2ego_f,ego2world_f,date_f,root_path):

    # lidar2ego_f = '/mnt/data/zlidar/lidar-sample/lidar2egos.pkl'
    # ego2world_f = '/mnt/data/zlidar/lidar-sample/ego2worlds.pkl'
    # date_f = '/mnt/data/zlidar/lidar-sample/date.pkl'
    lidar2ego_f = open(lidar2ego_f, 'rb')
    ego2world_f = open(ego2world_f, 'rb')
    date_f = open(date_f, 'rb')

    lidar2ego = pickle.load(lidar2ego_f)
    ego2world = pickle.load(ego2world_f)
    date_info = pickle.load(date_f)

    lidar2ego = lidar2ego.values
    ego2world = ego2world.values
    date_info = date_info.values

    lidar2ego_new = []
    ego2world_new = []
    date_new = []

    for index in range(lidar2ego.shape[0]-1):
        lidar2ego_new.append(np.reshape(lidar2ego[index+1],(4,4)))

    for index in range(ego2world.shape[0]-1):
        ego2world_new.append(np.reshape(ego2world[index+1],(4,4)))

    for index in range(date_info.shape[0]-1):
        date_new.append(date_info[index+1][0])
    
    assert len(date_new) == len(lidar2ego_new) == len(ego2world_new)

    lidar_info_file = open(os.path.join(root_path,'lidar_info','segment1.pkl'),'wb')
    ts_info_file = open(os.path.join(root_path,'ts_info','segment1.pkl'),'wb')
    ego_info_file = open(os.path.join(root_path,'ego_info','segment1.pkl'),'wb')

    pickle.dump(lidar2ego_new, lidar_info_file)
    pickle.dump(ego2world_new, ego_info_file)
    pickle.dump(date_new, ts_info_file)

    return None



if __name__ == '__main__':

    # lidar_path = '/mnt/data/zlidar/lidar-sample/lidar/'
    lidar2ego_f = '/mnt/data/zlidar/lidar-sample/lidar2egos.pkl'
    ego2world_f = '/mnt/data/zlidar/lidar-sample/ego2worlds.pkl'
    date_f = '/mnt/data/zlidar/lidar-sample/date.pkl'
    root_path = '/mnt/data/zlidar/lidar-sample/'

    # pc_all = splicing_lidar(lidar_path,lidar2ego_f,ego2world_f)
    # print(pc_all.shape)
    # #vis_pc(pc_all)
    # draw_scenes(pc_all, draw_origin=False)

    trans_data(lidar2ego_f,ego2world_f,date_f,root_path)



    

    

