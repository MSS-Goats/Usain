# STEP 1: Import the necessary modules.
from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(
    model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("bolt_raw/240.jpg")

cords = []
# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)
pose_landmarks_list = detection_result.pose_landmarks
for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]
    for landmark in pose_landmarks:
        print((landmark.x, landmark.y))
        cords.append((landmark.x * 800, landmark.y * 860))
        print((landmark.x * 800, landmark.y * 860))


image = Image.new("RGB", (1000, 1000), "white")

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto, 
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

res = draw_landmarks_on_image(image, detection_result)
image_np = Image.fromarray(res)
width, height = 1000, 1060
image = Image.new("RGB", (width, height), "white")

# Paste the NumPy array image onto the new image
image.paste(image_np, (0, 0))  # Example: paste at position (100, 100)

# Display the image
image.show()
# Save the image
image.save("numpy_image.png")
