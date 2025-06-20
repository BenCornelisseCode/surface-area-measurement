import cv2
import numpy as np
import cv2.aruco as aruco
# Import pytorch
import torch
torch.cuda.empty_cache()
import os
from segment_anything import SamPredictor, sam_model_registry

# Load the SAM2.0 model
sam_checkpoint = "/home/ben/OneDrive/Master Thesis/Internship/Code/sam_vit_b_01ec64.pth"  # Replace with the path to your SAM2.0 checkpoint
#sam_checkpoint = "/home/ben/OneDrive/Master Thesis/Internship/Code/sam_vit_h_4b8939.pth"
#sam_checkpoint = "/home/ben/OneDrive/Master Thesis/Internship/Code/sam_vit_l_0b3195.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
input_label = np.array([1])
masks, scores, logits = None, None, None
mask_number = 0  # Initialize mask number for cycling through masks

left_bottom = (0, 0)  # Left bottom calibaration point
left_top = (0, 180)  # Left top calibration point
right_bottom = (380, 0)  # Right bottom calibration point
right_top = (380, 180)  # Right top calibration point
expand_calibration_factor = 1

lower_color = np.array([0, 0, 0])
upper_color = np.array([179, 255, 255])
hsv_tolerance = np.array([10, 40, 40])

pixel_filter = 500 # Minimum pixel area to filter noise

# Load Calibration parameters
calibration_param = "/home/ben/OneDrive/Master Thesis/Internship/Code/camera_calibration_data.npz"
calibration_data = np.load(calibration_param)
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coefficients"]


# Initialize the camera
capture = cv2.VideoCapture(4)  # Open the camera (0 for default camera, 1 for external camera)

# Set the pixel format to MJPG
#capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set the resolution to 1920x1080

screen_width, screen_height = 1920, 1080

capture.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Verify the resolution
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a resizable window and set its size to match the camera resolution
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", width, height)

predicted = False  # Initialize the predicted mask
point_sam = None  # Initialize the point for SAM
picked_hsv = None
point = None
points = []
state = "calibration"  # State variable: "calibration" or "measurement"

transform_matrix = None  # Initialize the transformation matrix

# Sam Segment Functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# Mouse callback function to set points during calibration
def draw_circle(event, x, y, flags, param):
    global point, picked_hsv, lower_color, upper_color, point_sam,predicted,masks
    if state == "calibration":
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)

    elif state == "measurement":
        if event == cv2.EVENT_LBUTTONDOWN:
            point_sam = np.array([[int(x), int(y)]])
            predicted = False
            masks = None

# Calibration mode function, here 4 points are clicked to calibrate the camera
def calibration_mode(frame):
    global points, state, transform_matrix

    # --- ArUco marker detection ---
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # Only proceed if 4 markers are detected
    if ids is not None and len(ids) == 4:
        # Compute global centroid of all corners
        all_corners = np.concatenate([c[0] for c in corners], axis=0)
        global_center = all_corners.mean(axis=0)

        # For each marker, find the furthest corner from the global centroid
        marker_points = []
        for c in corners:
            dists = np.linalg.norm(c[0] - global_center, axis=1)
            furthest_idx = np.argmax(dists)
            furthest_corner = c[0][furthest_idx]
            marker_points.append(tuple(furthest_corner))
            # Visualize
            cv2.circle(frame, tuple(furthest_corner.astype(int)), 10, (0, 0, 255), 3)
        points = marker_points

        # Draw global center
        global_center_int = tuple(global_center.astype(int))
        cv2.circle(frame, global_center_int, 8, (255, 0, 255), 2)

        # Sort points to match: [left_bottom, left_top, right_bottom, right_top]
        # You may need to adjust this sorting logic for your marker layout!
        points_np = np.array(points)
        # Sort by y (top/bottom), then x (left/right)
        idx = np.lexsort((points_np[:,1], points_np[:,0]))
        sorted_points = points_np[idx]
        # Assign: bottom-left, top-left, bottom-right, top-right
        bl = sorted_points[0]
        tl = sorted_points[1]
        br = sorted_points[2]
        tr = sorted_points[3]
        src_points = np.array([bl, tl, br, tr], dtype=np.float32)

        # Undistort points
        #src_points_undist = cv2.undistortPoints(src_points.reshape(-1,1,2), camera_matrix, dist_coeffs, P=new_camera_matrix).reshape(-1,2)

        # Destination points (real-world)
        dst_points = np.array([left_bottom, left_top, right_bottom, right_top], dtype=np.float32)

        # Expand for your calibration factor if needed
        src_center = np.mean(src_points, axis=0)
        expanded_src_points = (src_points - src_center) * expand_calibration_factor + src_center
        dst_center = np.mean(dst_points, axis=0)
        expanded_dst_points = (dst_points - dst_center) * expand_calibration_factor + dst_center

        # Compute perspective transform
        transform_matrix = cv2.getPerspectiveTransform(expanded_src_points, expanded_dst_points)
        cv2.putText(frame, "Calibration Finished (Auto)", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        state = "measurement"
    else:
        cv2.putText(frame, "Show all 4 ArUco markers for auto-calibration", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 

# Here measurements are taken based on calibration and the picked color of the tentacle area
def measurement_mode(frame):
    global transform_matrix, points, state, point, lower_color, upper_color
    global predicted, point_sam, masks, scores, logits, mask_number


    if point_sam is not None and not predicted:
        # Predict masks from clicked point
        predictor.set_image(frame)
        masks, scores, logits = predictor.predict(
            point_coords=point_sam,
            point_labels=input_label,
            multimask_output=True,
        )
        predicted = True
        mask_number = 0  # Reset to show the first mask

    # Cycle through masks
    if predicted and masks is not None and key == ord('n'):
        mask_number = (mask_number + 1) % masks.shape[0]

    # Always display current mask if available
    if predicted and masks is not None:
        selected_mask = masks[mask_number]
        mask = selected_mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Warp the mask to top-down view using the perspective matrix
        a4_width_mm, a4_height_mm = right_top[0], right_top[1]
        warped_size = (int(a4_width_mm*expand_calibration_factor ), int(a4_height_mm*expand_calibration_factor))  # (width, height) in mm

        if transform_matrix is not None:
            warped_mask = cv2.warpPerspective(mask, transform_matrix, warped_size)


            # Count non-zero pixels in warped mask
            #warped_area_px = np.count_nonzero(warped_mask)
            warped_area_px = np.sum(warped_mask > 0)  # Count non-zero pixels

            # Since 1 pixel = 1 mm² in the warped top-down view
            mask_area_mm2 = warped_area_px
        else:
            mask_area_mm2 = 0

        # Overlay the mask on original frame for visualization
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame[:] = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)

        # Show clicked point
        if point_sam is not None:
            cv2.circle(frame, tuple(point_sam[0]), 5, (0, 255, 0), -1)

        # Display area in mm² in top-right corner
        area_text = f"Area: {mask_area_mm2:.2f} mm^2"
        cv2.rectangle(frame, (10, 10), (230, 45), (50, 50, 50), -1)
        cv2.putText(frame, area_text, (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Reset to calibration mode
    if key == ord('r'):
        points.clear()
        transform_matrix = None
        point = None
        state = "calibration"
        masks = None
        predicted = False
        point_sam = None

while True:
    ret, frame = capture.read()
    if not ret:
        break

    '''''
    center_x, center_y = width // 2, height // 2
    crop_width, crop_height = 1920, 1080
    x1 = center_x - crop_width // 2
    y1 = center_y - crop_height // 2
    x2 = center_x + crop_width // 2
    y2 = center_y + crop_height // 2
    frame = frame[y1:y2, x1:x2]
    '''

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image to the valid region of interest (optional)
    #x, y, w, h = roi
    #frame = undistorted_frame[y:y+h, x:x+w]
    #frame = undistorted_frame.copy()


    cv2.setMouseCallback("Camera", draw_circle, param=frame)

    key = cv2.waitKey(1)

    if state == "calibration":
        calibration_mode(frame)

    elif state == "measurement":
        measurement_mode(frame)
    
    # Display the frame
    cv2.imshow("Camera", frame)


    if key == 27:  # ESC key to exit
        break

capture.release()
cv2.destroyAllWindows()