import cv2
import numpy as np
import time

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
        if not isinstance(frame, np.ndarray):
            return None
        if frame.ndim != 3:
            return None
        frame = cv2.flip(frame, 1)

        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=2,
            minSize=(30, 30)
        )

        print("Faces:", len(faces), flush=True)

        display = frame.copy()

        if len(faces) == 0:
            cv2.putText(display, "NO FACE", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imwrite("/tmp/debug.png", display)
            print("Saved debug (no face)", flush=True)
            return None

        x_small, y_small, w_small, h_small = faces[0]

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        x = x_small * scale_x
        w = w_small * scale_x

        frame_w = frame.shape[1]
        face_center = x + w / 2

        if face_center < frame_w * 0.45:
            head = 0
        elif face_center > frame_w * 0.55:
            head = 2
        else:
            head = 1

        roi_gray = gray[y_small:y_small + h_small, x_small:x_small + w_small]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=1,
            minSize=(10, 10)
        )

        print("Eyes:", len(eyes), flush=True)

        eye = 1

        if len(eyes) >= 1:
            centers = []
            for (ex, ey, ew, eh) in eyes:
                centers.append(ex + ew / 2)

            eye_center = sum(centers) / len(centers)

            if eye_center < w_small * 0.45:
                eye = 0
            elif eye_center > w_small * 0.55:
                eye = 2
            else:
                eye = 1

        x_full = int(x_small * scale_x)
        y_full = int(y_small * scale_y)
        w_full = int(w_small * scale_x)
        h_full = int(h_small * scale_y)

        cv2.rectangle(display, (x_full, y_full), (x_full + w_full, y_full + h_full), (255, 0, 0), 2)

        cx = int(x_full + w_full / 2)
        cy = int(y_full + h_full / 2)
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

        cv2.line(display, (int(frame_w * 0.45), 0), (int(frame_w * 0.45), frame.shape[0]), (0,255,0), 2)
        cv2.line(display, (int(frame_w * 0.55), 0), (int(frame_w * 0.55), frame.shape[0]), (0,255,0), 2)

        for (ex, ey, ew, eh) in eyes:
            ex_full = int((x_small + ex) * scale_x)
            ey_full = int((y_small + ey) * scale_y)
            ew_full = int(ew * scale_x)
            eh_full = int(eh * scale_y)
            cv2.rectangle(display, (ex_full, ey_full), (ex_full + ew_full, ey_full + eh_full), (0, 255, 0), 2)

        cv2.putText(display, f"Head:{head} Eye:{eye}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # cv2.imwrite("/tmp/debug.png", display)
        # print("Saved debug image", flush=True)

        return {"head": head, "eye": eye}