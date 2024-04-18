from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
import os

base_options = python.BaseOptions(
    model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

image_width = 800
image_height = 860

POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])


def angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


image = mp.Image.create_from_file(os.getcwd() + "/bolt_raw/241.jpg")
coords = []
detection_result = detector.detect(image)
pose_landmarks_list = detection_result.pose_landmarks
for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]
    for landmark in pose_landmarks:
        # print((landmark.x, landmark.y))
        x_px = min(math.floor(landmark.x * image_width), image_width - 1)
        y_px = min(math.floor(landmark.y * image_height), image_height - 1)
        coords.append((x_px, y_px))
        # print((x_px, y_px))
for edge in POSE_CONNECTIONS:
    u = edge[0]
    v = edge[1]
    print(coords[u], coords[v], angle(coords[u], coords[v]))

# image.save(os.getcwd() + "/bolt_processed/own_drawing_" + filename)

# for filename in os.listdir(os.getcwd() + "/bolt_raw"):
#     image = mp.Image.create_from_file(os.getcwd() + "/bolt_raw/" + filename)

#     coords = []
#     detection_result = detector.detect(image)
#     pose_landmarks_list = detection_result.pose_landmarks
#     for idx in range(len(pose_landmarks_list)):
#         pose_landmarks = pose_landmarks_list[idx]
#         for landmark in pose_landmarks:
#             # print((landmark.x, landmark.y))
#             x_px = min(math.floor(landmark.x * image_width), image_width - 1)
#             y_px = min(math.floor(landmark.y * image_height), image_height - 1)
#             coords.append((x_px, y_px))
#             print((x_px, y_px))

#     image = Image.new("RGB", (1000, 1000), "white")

#     draw = ImageDraw.Draw(image)

#     for edge in POSE_CONNECTIONS:
#         u = edge[0]
#         v = edge[1]
#         draw.line([coords[u], coords[v]], fill="black", width=2)

#     image.save(os.getcwd() + "/bolt_processed/own_drawing_" + filename)
