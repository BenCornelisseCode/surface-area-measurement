import cv2
import time
import os
from datetime import datetime

def capture_frames(duration_sec, output_dir_base="captured_frames"):
    # Create a unique output directory with date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Open the default camera
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Save frame as JPG file
        filename = os.path.join(output_dir, f"{frame_count}.jpg")
        cv2.imwrite(filename, frame)
        frame_count += 1

        # Check if duration has passed
        if time.time() - start_time > duration_sec:
            break

    cap.release()
    print(f"Saved {frame_count} frames to '{output_dir}'.")

if __name__ == "__main__":
    duration = float(input("Enter duration in seconds: "))
    capture_frames(duration)