import numpy as np
from filterpy.kalman import KalmanFilter
from enum import Enum

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker:
    count = 1

    def __init__(self, y, R, wx, wy, vmax, w, h, dt=1/30, lambda_=0.05, ema_alpha=0.9, shape_weight=0.0):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        self.kf.F = np.array([
            [1, dt, 0.5 * dt * dt, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt * dt],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        self.kf.R = R
        self.kf.P = np.zeros((6, 6))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))

        G = np.zeros((6, 2))
        G[0, 0] = 0.5 * dt * dt
        G[1, 0] = dt
        G[2, 0] = 1
        G[3, 1] = 0.5 * dt * dt
        G[4, 1] = dt
        G[5, 1] = 1
        Q0 = np.array([[wx, 0], [0, wy]])
        self.base_Q = np.dot(np.dot(G, Q0), G.T)
        self.kf.Q = self.base_Q.copy()

        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = 0
        self.kf.x[3] = y[1]
        self.kf.x[4] = 0
        self.kf.x[5] = 0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h
        self.status = TrackStatus.Tentative
        self.lambda_ = lambda_
        self.ema_alpha = ema_alpha
        self.r_squared_ema = 0
        self.shape_weight = shape_weight  # aspect-ratio gate weight (0 = disabled)

    def update(self, y, R):
        residual = y - np.dot(self.kf.H, self.kf.x)
        r_norm_sq = np.linalg.norm(residual)**2
        self.r_squared_ema = self.ema_alpha * self.r_squared_ema + (1 - self.ema_alpha) * r_norm_sq
        alpha = 1 + self.lambda_ * self.r_squared_ema
        self.kf.Q = alpha * self.base_Q
        self.kf.update(y, R)

    def predict(self):
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x

    def distance(self, y, R, det_w=None, det_h=None):
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S))
        cost = mahalanobis[0,0] + logdet

        # ── shape gate (aspect-ratio prior; 0 = disabled → identical to UAVC) ──
        # GLD trains geometry-aligned features; the vessel aspect ratio is the
        # cheapest bbox-only proxy for that shape prior. Penalise track↔det
        # matches whose log aspect-ratio differs. Scale-invariant, NPU-free.
        if self.shape_weight > 0.0 and det_w is not None and det_h is not None \
                and det_h > 0 and self.h > 0:
            trk_ar = self.w / max(self.h, 1e-6)
            det_ar = det_w / max(det_h, 1e-6)
            shape_d = abs(np.log(max(det_ar, 1e-6) / max(trk_ar, 1e-6)))
            cost = cost + self.shape_weight * shape_d
        return cost