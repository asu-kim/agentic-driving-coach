import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def process(self, frame):
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]

        # =========================
        # HEAD (based on face position)
        # =========================
        frame_w = frame.shape[1]
        face_center = x + w / 2

        if face_center < frame_w * 0.4:
            head = 4   # left
        elif face_center > frame_w * 0.6:
            head = 3   # right
        else:
            head = 5   # center

        # =========================
        # EYE (based on eye position)
        # =========================
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)

        eye = 8  # default center

        if len(eyes) >= 2:
            ex, ey, ew, eh = eyes[0]
            eye_center = ex + ew / 2

            if eye_center < w * 0.4:
                eye = 6   # looking left
            elif eye_center > w * 0.6:
                eye = 7   # looking right
            else:
                eye = 8   # center

        return {"head": head, "eye": eye}