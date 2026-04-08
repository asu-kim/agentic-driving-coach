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

        # -------- SAFETY --------
        if not isinstance(frame, np.ndarray):
            return None
        if frame.ndim != 3:
            return None

        # -------- RESIZE (VERY IMPORTANT) --------
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # -------- FACE DETECTION (TUNED) --------
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(50, 50)
        )

        print("Faces:", len(faces), flush=True)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]

        # -------- SCALE BACK --------
        scale_x = frame.shape[1] / 320
        frame_w = frame.shape[1]

        face_center = (x + w / 2) * scale_x

        # -------- HEAD --------
        if face_center < frame_w * 0.4:
            head = 4
        elif face_center > frame_w * 0.6:
            head = 3
        else:
            head = 5

        # -------- EYE DETECTION --------
        roi_gray = gray[y:y+h, x:x+w]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(15, 15)
        )

        print("Eyes:", len(eyes), flush=True)

        eye = 8

        if len(eyes) >= 1:
            ex, ey, ew, eh = eyes[0]
            eye_center = ex + ew / 2

            if eye_center < w * 0.4:
                eye = 6
            elif eye_center > w * 0.6:
                eye = 7
            else:
                eye = 8

        return {"head": head, "eye": eye}