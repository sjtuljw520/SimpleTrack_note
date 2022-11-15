import matplotlib.pyplot as plt, numpy as np
from ..data_protos import BBox


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8), vc=None):
        self.figure = plt.figure(name, figsize=figsize)
        plt.axis('equal')
        plt.xlim((vc[0][0]-76, vc[0][0]+76))
        plt.ylim((vc[0][1]-76, vc[0][1]+76))
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256
        }
        self.BOX_COLOR = [
            np.array([191, 4, 54]) / 256,
            np.array([224, 133, 250]) / 256,
            np.array([32, 64, 40]) / 256, 
            np.array([25, 25, 112])/ 256,
            np.array([0, 0, 205])/ 256,
            np.array([0, 206, 209])/ 256,
            np.array([32, 178, 170])/ 256,
            np.array([34, 139, 34])/ 256,
            np.array([188, 143, 143])/ 256,
            np.array([255, 255, 0])/ 256,
            np.array([165, 42, 42])/ 256,
            np.array([255, 20, 147])/ 256,
            np.array([208, 32, 144])/ 256,
            np.array([74, 112, 139])/ 256,
            np.array([255, 48, 48])/ 256,
            np.array([255, 165, 0])/ 256,
            np.array([139, 117, 0])/ 256,
            np.array([205, 205, 0])/ 256,
            np.array([139, 101, 8])/ 256,
            np.array([102, 205, 170])/ 256,
        ]
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, color='gray'):
        vis_pc = np.asarray(pc)
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=self.COLOR_MAP[color], s=0.01)
    
    def handler_box_gt(self, box: BBox, message: str='', color='red', linestyle='solid'):
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        corner_index = np.random.randint(0, 4, 1)
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color])

    def handler_box(self, box: BBox, message: str='', color_id=0, linestyle='solid'):
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.BOX_COLOR[int(color_id)%20], linestyle=linestyle)
        corner_index = np.random.randint(0, 4, 1)
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.BOX_COLOR[int(color_id)%20])