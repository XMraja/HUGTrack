from .mapper import Mapper
import numpy as np
import cv2
import csv

# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)
        

    def get_box(self):
        return [self.bb_left, self.bb_top, self.bb_width, self.bb_height]


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# # -------- 只用“中心点”的轻量位姿估计器 --------
# class PoseEstimator:
#     def __init__(self, max_points=10, alpha=0.35, max_angle_deg=5.0, max_trans_px=5.0, min_pts=3):
#         self.max_points = int(max_points)
#         self.alpha = float(alpha)
#         self.max_angle = np.deg2rad(max_angle_deg)
#         self.max_trans = float(max_trans_px)
#         self.min_pts = int(min_pts)

#         # 平滑状态
#         self._theta_prev = 0.0
#         self._T_prev = np.zeros((2, 1), dtype=np.float64)

#     @staticmethod
#     def _centers(boxes):
#         # boxes: list[(x,y,w,h)] -> Nx2 float32
#         return np.array([[x + 0.5 * w, y + h] for (x, y, w, h) in boxes], dtype=np.float32)

#     def update(self, prev_boxes, curr_boxes):
#         """
#         prev_boxes, curr_boxes: list[(x,y,w,h)]
#         返回 (delta_R(3x3), delta_T(3x1)), 若点数不足或越阈值则返回单位增量
#         """
#         if len(prev_boxes) < self.min_pts or len(curr_boxes) < self.min_pts:
#             return np.eye(3), np.zeros((3, 1), dtype=np.float64)

#         # 只取最多 max_points（按“当前帧置信度排序后”在外部截断）
#         k = min(self.max_points, len(prev_boxes), len(curr_boxes))
#         P = self._centers(prev_boxes[:k])
#         Q = self._centers(curr_boxes[:k])

#         # 至少 3 个点才能拟合
#         if P.shape[0] < self.min_pts:
#             return np.eye(3), np.zeros((3, 1), dtype=np.float64)

#         M, inliers = cv2.estimateAffinePartial2D(
#             P, Q, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=300, confidence=0.99
#         )
#         if M is None:
#             return np.eye(3), np.zeros((3, 1), dtype=np.float64)

#         # 分解旋转和平移（只取 z 轴小角度）
#         theta = np.arctan2(M[1, 0], M[0, 0])
#         T = np.array([[M[0, 2]], [M[1, 2]]], dtype=np.float64)

#         # 阈值：过大就不更新（返回单位增量）
#         if abs(theta) > self.max_angle or float(np.linalg.norm(T)) > self.max_trans:
#             return np.eye(3), np.zeros((3, 1), dtype=np.float64)

#         # 平滑
#         theta_s = self.alpha * theta + (1.0 - self.alpha) * self._theta_prev
#         T_s = self.alpha * T + (1.0 - self.alpha) * self._T_prev

#         # 保存平滑状态
#         self._theta_prev = theta_s
#         self._T_prev = T_s

#         # 构造 3x3 ΔR、3x1 ΔT
#         R = np.eye(3, dtype=np.float64)
#         c, s = np.cos(theta_s), np.sin(theta_s)
#         R[0, 0] = c; R[0, 1] = -s
#         R[1, 0] = s; R[1, 1] =  c
#         T3 = np.array([[T_s[0, 0]], [T_s[1, 0]], [0.0]], dtype=np.float64)

#         return R, T3


# Detector类，用于从文本文件读取任意一帧中的目标检测的结果
class Detector:
    def __init__(self, add_noise = False):
        self.seq_length = 0
        self.gmc = None
        self.add_noise = add_noise
        # # 中心点 CMC：最多10点，数量不足不更新，平滑+阈值
        # self.pose_est = PoseEstimator(
        #     max_points=10, alpha=0.35, max_angle_deg=5.0, max_trans_px=5.0, min_pts=3
        # )
        self.pose_delta = {}   # frame -> (theta_rad, tx_px, ty_px)
        # self.use_offline_pose = False


    def load(self,cam_para_file, det_file, pose_delta_file=None):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.pose_delta.clear()
        if pose_delta_file is not None:
            with open(pose_delta_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row: 
                        continue
                    frame = int(row["frame"])
                    theta = float(row["theta_rad"])
                    self.pose_delta[frame] = theta
        if self.pose_delta is None:
        # 基线：使用原始稳定版
            self.load_detfile_base(det_file)
        else:
            self.load_detfile(det_file)

    def load_detfile(self, filename):
        self.dets = {}
        self.prev_boxes_for_pose = None  # 上一帧用于外参估计的 box 列表
        self.seq_length = 0

        # —— 帧缓冲：先收集同一帧的所有 det，帧切换时统一处理 —— #
        cur_frame_id = None
        frame_dets = []

        def apply_offline_rotation(frame_id):
            """仅应用 ΔR（若无则跳过），不使用 ΔT"""
            # if not self.use_offline_pose:
            #     return
            theta = self.pose_delta.get(frame_id, 0.0)
            if abs(theta) < 1e-6:
                return
            # 小保护：可按需裁剪极端值
            # theta = np.clip(theta, -np.deg2rad(8.0), np.deg2rad(8.0))
            # print("loading")
            c, s = np.cos(theta), np.sin(theta)
            dR = np.eye(3, dtype=np.float64)
            dR[0, 0] = c; dR[0, 1] = -s
            dR[1, 0] = s; dR[1, 1] =  c
            self.mapper.apply_delta_pose(delta_R=dR, delta_T=np.zeros((3,1)))
            # print(f"Applying ΔR at frame {frame_id}: {np.rad2deg(theta):.6f} deg")


        def process_one_frame(frame_id, dets_this_frame):
           
            apply_offline_rotation(frame_id)

            for det in dets_this_frame:
                if self.add_noise:
                    noise_z = (0.5 / 180.0 * np.pi) if (frame_id % 2 == 0) else (-0.5 / 180.0 * np.pi)
                    self.mapper.disturb_campara(noise_z)

                det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])

                if self.add_noise:
                    self.mapper.reset_campara()
            self.dets[frame_id] = dets_this_frame


        # 打开文本文件filename
        with open(filename, 'r') as f:
            # 读取文件中的每一行
            for line in f.readlines():
                # 将每一行的内容按照空格分开
                line = line.strip().split(',')
                frame_id = int(line[0])
                if frame_id > self.seq_length:
                    self.seq_length = frame_id
                det_id = int(line[1])
                
                # 检测到新帧，先处理上一帧
                if cur_frame_id is not None and frame_id != cur_frame_id:
                    process_one_frame(cur_frame_id, frame_dets)
                    frame_dets = []
                # 新建一个Detection对象
                det = Detection(det_id)
                det.bb_left = float(line[2])
                det.bb_top = float(line[3])
                det.bb_width = float(line[4])
                det.bb_height = float(line[5])
                det.conf = float(line[6])
                det.det_class = int(line[7])
                if det.det_class == -1:
                    det.det_class = 0
                frame_dets.append(det)
                cur_frame_id = frame_id
                
        # 文件结束别忘了处理最后一帧
        if cur_frame_id is not None and len(frame_dets) > 0:
            process_one_frame(cur_frame_id, frame_dets)

    def load_detfile_base(self, filename):
        self.dets = dict()
        # 打开文本文件filename
        with open(filename, 'r') as f:
            # 读取文件中的每一行
            for line in f.readlines():
                # 将每一行的内容按照空格分开
                line = line.strip().split(',')
                frame_id = int(line[0])
                if frame_id > self.seq_length:
                    self.seq_length = frame_id
                det_id = int(line[1])
                # 新建一个Detection对象
                det = Detection(det_id)
                det.bb_left = float(line[2])
                det.bb_top = float(line[3])
                det.bb_width = float(line[4])
                det.bb_height = float(line[5])
                det.conf = float(line[6])
                det.det_class = int(line[7])
                if det.det_class == -1:
                    det.det_class = 0
                
                if self.add_noise:
                    if frame_id % 2 == 0:
                        noise_z = 0.5/180.0*np.pi
                    else:
                        noise_z = -0.5/180.0*np.pi
                    self.mapper.disturb_campara(noise_z)

                det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
                
                if self.add_noise:
                    self.mapper.reset_campara()

                # 将det添加到字典中
                if frame_id not in self.dets:
                    self.dets[frame_id] = []
                self.dets[frame_id].append(det)
                

    def get_dets(self, frame_id,conf_thresh = 0,det_class = 0):
        # dets = self.dets[frame_id]
        dets = self.dets.get(frame_id, [])
        dets = [det for det in dets if det.det_class == det_class and det.conf >= conf_thresh]
        return dets
    
    
    def cmc(self,x,y,w,h,frame_id):
        u,v = self.mapper.xy2uv(x,y)
        affine = self.gmc.get_affine(frame_id)
        M = affine[:,:2]
        T = np.zeros((2,1))
        T[0,0] = affine[0,2]
        T[1,0] = affine[1,2]

        p_center = np.array([[u],[v-h/2]])
        p_wh = np.array([[w],[h]])
        p_center = np.dot(M,p_center) + T
        p_wh = np.dot(M,p_wh)

        u = p_center[0,0]
        v = p_center[1,0]+p_wh[1,0]/2

        xy,_ = self.mapper.uv2xy(np.array([[u],[v]]),np.eye(2))

        return xy[0,0],xy[1,0]


