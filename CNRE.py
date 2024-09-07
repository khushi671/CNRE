import cv2
import numpy as np
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from tkinter import *
from tkinter import filedialog, simpledialog, scrolledtext
import pickle
import os
import random
import ctypes

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = np.hypot(cx - pt[0], cy - pt[1])
                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

pickled_img = None
file = None
salt = None
key = None
coordinates = []

def detect_car(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    if 1.0 < aspect_ratio < 4.0 and area > 1000:
        return (x, y, w, h)
    return None

def adjust_coordinates(x, y, w, h):
    adjusted_x = x + 10
    adjusted_y = y + 10
    return adjusted_x, adjusted_y, w, h

def format_coordinates(x, y):
    symbol = '*'
    num_symbols_x = min(int(x / 100), 10)
    num_symbols_y = min(int(y / 100), 10)
    return f'{symbol * num_symbols_x}X{x:04d}{symbol * num_symbols_y}Y{y:04d}'

def clip_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = random.randint(0, total_frames // 2)
    end_frame = random.randint(start_frame + int(fps) * 5, min(start_frame + int(fps) * 30, total_frames - 1))

    output_dir = os.path.dirname(file)
    output_file = os.path.join(output_dir, "clipped_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    return output_file

def generate_jaadu(video_path):
    global coordinates
    tracker = EuclideanDistTracker()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return ""

    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    car_detected = False
    frames_without_car = 0
    max_frames_without_car = 30
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[140:720, 380:700]
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                car_box = detect_car(cnt)
                if car_box:
                    x, y, w, h = car_box
                    x, y, w, h = adjust_coordinates(x, y, w, h)
                    detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            coordinates.append(format_coordinates(x + w // 2, y + h // 2))
            car_detected = True
            frames_without_car = 0

        if not detections:
            frames_without_car += 1

        frame_counter += 1

        if car_detected and frames_without_car > max_frames_without_car:
            break

        if frame_counter > 500:
            break

    cap.release()

    if not coordinates:
        return ""

    coords_string = ''.join(coordinates)
    return coords_string

def init_encryption():
    device_password = 'Curiosity@123'
    password = simpledialog.askstring("Password", "Enter system password:", initialvalue=None, show='*')
    if password != device_password:
        print("Wrong Password")
        return

    select_file()

    global salt, key, file
    file_path = filedialog.askopenfile(mode='rb', filetype=[('Video file', '*.mp4')])
    if file_path is None:
        return

    file_path = file_path.name
    clipped_video_path = clip_video(file_path)

    if not clipped_video_path:
        return

    salt = generate_jaadu(clipped_video_path)
    if not salt:
        return
        
    String_rand = b'Hello World'

    key = PBKDF2(String_rand, salt, dkLen=32)
    encrypt_file()

def select_file():
    global pickled_img, file

    file1 = filedialog.askopenfile(mode='rb', filetype=[('image file', '*.jpg, *.png')])
    if file1 is not None:
        image = file1.read()
        file = file1.name
        file1.close()
        pickled_img = pickle.dumps(image)
        output_text.insert(END, "File selected successfully!\n")
        output_text.insert(END, "Please wait...\n")
    output_text.insert(END, "You can proceed!\n")

def encrypt_file():
    if pickled_img is not None:
        message = pickled_img
        cipher = AES.new(key, AES.MODE_CBC)
        cipher_data = cipher.encrypt(pad(message, AES.block_size))

        with open(file, 'wb') as f:
            f.write(cipher.iv)
            f.write(cipher_data)
        output_text.insert(END, "Image encrypted successfully!\n")
    else:
        output_text.insert(END, "Select a File!\n")

def is_caps_lock_on():
    # On Windows, use ctypes to check Caps Lock status
    hll_dll = ctypes.WinDLL("User32.dll")
    VK_CAPITAL = 0x14
    return hll_dll.GetKeyState(VK_CAPITAL) & 0xffff != 0

def decrypt_file():
    global file

    device_password = 'Curiosity@123'
    password = simpledialog.askstring("Password", "Enter system password:", initialvalue=None, show='*')
    if password != device_password:
        print("Wrong password")
        return
    if not password:
        print("Password not provided.")
        return

    select_file()

    if file is None:
        print("No file selected for decryption.")
        return

    if file is not None:
        with open(file, 'rb') as f:
            iv = f.read(16)
            decrypt_data = f.read()

        cipher = AES.new(key, AES.MODE_CBC, iv=iv)
        try:
            original = unpad(cipher.decrypt(decrypt_data), AES.block_size)
            unpickled_img = pickle.loads(original)

            with open(file, 'wb') as f:
                f.write(unpickled_img)
            output_text.insert(END, "Image decrypted successfully!\n")
            print("Image decrypted successfully!")
        except ValueError as e:
            print(f"Decryption failed: {e}")
            output_text.insert(END, "Decryption failed! Incorrect padding.\n")
    else:
        print("No files Selected!")


root = Tk()
root.title("Encryption and Decryption Tool")
root.geometry("600x400")

frame = Frame(root)
frame.pack(pady=20)

# Add icon files
encrypt_icon = PhotoImage(file='encrypt.png')
decrypt_icon = PhotoImage(file='decrypt.png')

output_text = scrolledtext.ScrolledText(root, width=60, height=10)
output_text.pack(pady=20)

encrypt_button = Button(frame, text="Encrypt", command=init_encryption, compound=LEFT, image=encrypt_icon)
encrypt_button.pack(padx=10, pady=10)

decrypt_button = Button(frame, text="Decrypt", command=decrypt_file, compound=LEFT, image=decrypt_icon)
decrypt_button.pack(padx=10, pady=10)

if is_caps_lock_on():
    output_text.insert(END, "Warning: Caps Lock is ON!\n")

root.mainloop()
