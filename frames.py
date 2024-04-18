import cv2
import os


def get_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0

    # Read until the video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image file
            frame_path = os.path.join(output_dir, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Increment frame count
            frame_count += 1
            print(frame_count)
        else:
            break

    # Release the video capture object
    cap.release()


get_frames(os.getcwd() + "/vids/drivephase.mp4", "frames")
