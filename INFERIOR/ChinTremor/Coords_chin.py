"""
Coords_chin.py

Python file intended to obtain the coordinates of keypoints of the chin to measure the tremor and the amplitude
of the mouth during a video. It uses YOLO for face detection and Mediapipe for a second face detection and
keypoint characterization.
This code is generalized for any case where the final intention of the whole process is the comparison of several points
with one used as a reference point.

Last edition: 24/07/2024
Author: Javier Hernández Rubia
"""



import mediapipe as mp
import cv2
import numpy as np
import argparse
from warnings import filterwarnings
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
import torch
import json
import matplotlib.pyplot as plt


device = torch.device("cpu") # CPU used as default to avoid GPU usage.
print(f'Using device: {device}')

# Setting up Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# This needs to be changed according to the location of the model
yolo_model = '/home/jhernandez2020/Escritorio/Facial_Mooveo/yolov8n-face.pt'


def face_roi_yolo(model_face, conf_thresh, frame, offset):
    """
    First facial detection. It will return a ROI corresponding to facial coordinates.
    """
    try:
        prediction = model_face.track(frame, persist=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Switching to CPU...")
            model_face.to("cpu")
            prediction = model_face.track(frame, persist=True)
        else:
            raise e

    bboxes = prediction[0].boxes
    face_roi = []

    if bboxes.shape[0] > 0:
        for i in range(bboxes.shape[0]):
            if bboxes[i].conf.numpy()[0] > conf_thresh:
                # print(bboxes.id.numpy().astype(int))
                bbox_pred = bboxes[i].xyxy.numpy()
                bbox_pred = bbox_pred.astype(int)
                bbox_pred[0][:2] -= offset
                bbox_pred[0][2:] += offset
                bbox_pred[0][::2] = bbox_pred[0][::2].clip(min=0, max=frame.shape[1])
                bbox_pred[0][1::2] = bbox_pred[0][1::2].clip(min=0, max=frame.shape[0])
                face_roi.append(bbox_pred[0])
    else:
        face_roi = [None]

    return face_roi

def import_yolo_detection(yolo_model):
    model = YOLO(yolo_model).to(device)
    return model

def export_coordinates(image, face_landmarks):
    frame_eye_dict = {'Ref': [], 'Chin': []}
    for idx, landmark in enumerate(face_landmarks):
        if idx in chin_points:
            frame_eye_dict['Chin'].append((landmark.x, landmark.y, landmark.z))
            # 17 corresponds to the reference point used for this specific application. It can be changed as desired
            frame_eye_dict['Ref'].append((face_landmarks[0].x, face_landmarks[0].y, face_landmarks[0].z))
    return frame_eye_dict

face_detector = import_yolo_detection(yolo_model)

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True)
parser.add_argument('-f', '--file_name', required=True)
args = parser.parse_args()
video_path = args.video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
bboxes_frames = []


chin_points = [18, 200, 199, 175]

colors_bbox = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       min_face_detection_confidence=0.7,
                                       min_tracking_confidence=0.9)
detector = vision.FaceLandmarker.create_from_options(options)

coords_dict = {}

fc = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_copy = frame.copy()
        h_whole, w_whole, _ = frame.shape

        face_roi = face_roi_yolo(face_detector, 0.5, frame, 20)
        for i in range(len(face_roi)):
            if face_roi[i] is not None:
                xmin, ymin, xmax, ymax = face_roi[i]
                bboxes_frames.append(face_roi[i])
            else:
                xmin, ymin, xmax, ymax = bboxes_frames[-1]

            frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors_bbox[i], 4)

            face_cropped = frame_rgb[int(ymin):int(ymax), int(xmin):int(xmax)]
            face_cropped = np.array(face_cropped, dtype=np.uint8)
            face_cropped = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_cropped)
            detection_result = detector.detect(face_cropped)

            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    coords_dict[f'Frame{fc}'] = export_coordinates(face_cropped.numpy_view(), face_landmarks)
                fc += 1
        print(fc)
    else:
        break
file_path = args.file_name

# Open the file in write mode and use json.dump() to save the dictionary
with open(file_path, 'w') as json_file:
    json.dump(coords_dict, json_file, indent=4)

print(f'Dictionary saved as JSON in {file_path}')