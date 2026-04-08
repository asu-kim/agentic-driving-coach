import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.yaw_old = None

    def process(self, frame):
        if frame is None:
            return None

        frame = np.ascontiguousarray(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        face = res.multi_face_landmarks[0].landmark
        px = [(lm.x * w, lm.y * h) for lm in face]

        if len(px) < 363:
            return None

        # -------- HEAD --------
        image_points = np.array(
            [px[1], px[152], px[33], px[263], px[61], px[291]],
            dtype=np.float64
        )

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1)
            ],
            dtype=np.float64
        )

        focal_length = float(w)
        center = (w/2.0, h/2.0)

        camera_mat = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]],
            dtype=np.float64
        )

        dist_coeff = np.zeros((4,1), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_mat,
            dist_coeff
        )

        head = 5
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))

            if self.yaw_old is None:
                self.yaw_old = yaw
            else:
                self.yaw_old = 0.8*self.yaw_old + 0.2*yaw

            if self.yaw_old > 15:
                head = 3
            elif self.yaw_old < -15:
                head = 4

        # -------- EYE --------
        def clamp(x):
            return max(0.0, min(1.0, x))

        if len(px) > 477:
            rx = sum(px[i][0] for i in [468,469,470,471,472]) / 5.0
            lx = sum(px[i][0] for i in [473,474,475,476,477]) / 5.0
        else:
            lx = 0.5 * (px[33][0] + px[133][0])
            rx = 0.5 * (px[362][0] + px[263][0])

        l_ratio = clamp((lx - px[33][0]) / max(px[133][0]-px[33][0], 1))
        r_ratio = clamp((rx - px[362][0]) / max(px[263][0]-px[362][0], 1))

        gaze = 0.5 * (l_ratio + r_ratio)

        eye = 8
        if gaze < 0.40:
            eye = 6
        elif gaze > 0.60:
            eye = 7

        return {"head": head, "eye": eye}