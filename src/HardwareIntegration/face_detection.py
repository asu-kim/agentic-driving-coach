import cv2
import mediapipe as mp
import socket
import pickle
import struct

HOST = "10.218.100.78"
PORT = 15045

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print("Waiting for LF connection...")
conn, addr = sock.accept()
print("Connected")

mp_face = mp.solutions.face_mesh.FaceMesh()

def recv_full(conn, size):
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

while True:
    try:
        # read frame length
        raw_len = recv_full(conn, 4)
        if not raw_len:
            continue
        msg_len = struct.unpack(">I", raw_len)[0]

        # read full frame
        data = recv_full(conn, msg_len)
        if not data:
            continue

        frame = pickle.loads(data)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if not res.multi_face_landmarks:
            out = None
        else:
            h, w = frame.shape[:2]
            face = res.multi_face_landmarks[0].landmark
            out = [(lm.x * w, lm.y * h) for lm in face]

        payload = pickle.dumps(out)
        conn.sendall(struct.pack(">I", len(payload)) + payload)

    except Exception as e:
        print("Worker error:", e)