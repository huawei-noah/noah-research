import numpy as np
import pyflex

class Camera():
    def __init__(self, screenWidth, screenHeight):
        
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        
        self.num_cameras = 4
        self.camera_view = None
        
        self.cam_dis = 6.
        self.cam_height = 10.
        self.cam_deg = np.array([0., 90., 180., 270.]) + 45.
    
    def set_init_camera(self, camera_view):
        self.camera_view = camera_view
        
        if self.camera_view == 0: # top view
            self.camPos = np.array([0., self.cam_height+10., 0.])
            self.camAngle = np.array([0., -np.deg2rad(90.), 0.])
        elif self.camera_view == 1:
            self.camPos = np.array([self.cam_dis, self.cam_height, self.cam_dis])
            self.camAngle = np.array([np.deg2rad(self.cam_deg[0]), -np.deg2rad(45.), 0.])
        elif self.camera_view == 2:
            self.camPos = np.array([self.cam_dis, self.cam_height, -self.cam_dis])
            self.camAngle = np.array([np.deg2rad(self.cam_deg[1]), -np.deg2rad(45.), 0.])
        elif self.camera_view == 3:
            self.camPos = np.array([-self.cam_dis, self.cam_height, -self.cam_dis])
            self.camAngle = np.array([np.deg2rad(self.cam_deg[2]), -np.deg2rad(45.), 0.])
        elif self.camera_view == 4:
            self.camPos = np.array([-self.cam_dis, self.cam_height, self.cam_dis])
            self.camAngle = np.array([np.deg2rad(self.cam_deg[3]), -np.deg2rad(45.), 0.])
        else:
            raise ValueError('camera_view not defined')
        
        # set camera
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
    
    def init_multiview_cameras(self):
        self.camPos_list, self.camAngle_list = [], []
        self.cam_x_list = np.array([self.cam_dis, self.cam_dis, -self.cam_dis, -self.cam_dis])
        self.cam_z_list = np.array([self.cam_dis, -self.cam_dis, -self.cam_dis, self.cam_dis])
        
        self.rad_list = np.deg2rad(self.cam_deg)
        for i in range(self.num_cameras):
            self.camPos_list.append(np.array([self.cam_x_list[i], self.cam_height, self.cam_z_list[i]]))
            self.camAngle_list.append(np.array([self.rad_list[i], -np.deg2rad(45.), 0.]))
        
        self.cam_intrinsic_params = np.zeros([len(self.camPos_list), 4]) # [fx, fy, cx, cy]
        self.cam_extrinsic_matrix = np.zeros([len(self.camPos_list), 4, 4]) # [R, t]
        
        return self.camPos_list, self.camAngle_list, self.cam_intrinsic_params, self.cam_extrinsic_matrix
        
    def get_cam_params(self):
        # camera intrinsic parameters
        projMat = pyflex.get_projMatrix().reshape(4, 4).T 
        cx = self.screenWidth / 2.0
        cy = self.screenHeight / 2.0
        fx = projMat[0, 0] * cx
        fy = projMat[1, 1] * cy
        camera_intrinsic_params = np.array([fx, fy, cx, cy])

        # camera extrinsic parameters
        cam_extrinsic_matrix = pyflex.get_viewMatrix().reshape(4, 4).T
        
        return camera_intrinsic_params, cam_extrinsic_matrix
    