#!/usr/bin/env python

import cv2
import numpy as np
import os

# Defining the dimensions of the checkerboard
CHECKERBOARD = (12, 18)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vectors to store 3D points and 2D points for each checkerboard image
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Defining the world coordinates for 3D points
square_size = 12.43 # size of a square in mm (e.g., 25mm = 2.5cm)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Initialize webcam
capture = cv2.VideoCapture(4)  # Use 0 for the default webcam
#capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set the resolution to 1920x1080
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify the resolution
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a resizable window and set its size to match the camera resolution
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", width, height)

# Variables to store captured images
captured_images = []
max_images = 30  # Number of images to capture

while len(captured_images) < max_images:
    ret, frame = capture.read()
    if not ret:
        break

    '''

    center_x, center_y = width // 2, height // 2
    crop_width, crop_height = 1920, 1080
    x1 = center_x - crop_width // 2
    y1 = center_y - crop_height // 2
    x2 = center_x + crop_width // 2
    y2 = center_y + crop_height // 2
    frame = frame[y1:y2, x1:x2]
    '''
    
    # Display the live feed
    cv2.putText(frame, f"Images Captured: {len(captured_images)}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'n' to capture an image", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    key = cv2.waitKey(1)
    if key == ord('n'):
        # Capture the current frame
        captured_images.append(frame.copy())

    cv2.imshow("Camera", frame)

    if key == 27:  # Press 'ESC' to exit
        break

capture.release()
cv2.destroyAllWindows()

if len(captured_images) < max_images:
    print("Not enough images captured for calibration. Exiting...")
    exit()


# Process the captured images for calibration
for img, i in zip(captured_images, range(len(captured_images))):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        # Refine pixel coordinates for the corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img_with_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
        cv2.imshow("Checkerboard Detection", img_with_corners)
        print(f"Checkerboard found in image {i+1}")
        cv2.waitKey(200)
    else:
        # Show the image where the checkerboard was not found
        cv2.imshow("Checkerboard Detection - Not Found", img)
        print(f"Checkerboard not found in image {i+1}")
        cv2.waitKey(200)

cv2.destroyAllWindows()

if len(objpoints) < max_images:
    print("Not all images had detectable checkerboards. Calibration may be inaccurate.")

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration results to a file
calibration_data = {
    "camera_matrix": mtx,
    "dist_coefficients": dist,
    "rotation_vectors": rvecs,
    "translation_vectors": tvecs
}

# Save using numpy
np.savez("camera_calibration_data.npz", **calibration_data)

# Print confirmation
print("Calibration data saved to 'camera_calibration_data.npz'")