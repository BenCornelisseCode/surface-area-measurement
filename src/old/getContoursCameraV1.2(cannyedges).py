import cv2
import numpy as np

distance_real = 30.0 # Real-world distance in cm (for calibration)

left_bottom = (0, 0)  # Left bottom calibaration point
left_top = (0, 21.0)  # Left top calibration point
right_bottom = (29.7, 0)  # Right bottom calibration point
right_top = (29.7, 21.0)  # Right top calibration point


lower_color = np.array([0, 0, 0])
upper_color = np.array([179, 255, 255])
hsv_tolerance = np.array([8, 60, 30])

pixel_filter = 100 # Minimum pixel area to filter noise
alpha = 0  # Smoothing factor (0.0 = no smoothing, 1.0 = only new frame)

capture = cv2.VideoCapture(4)  # Open the camera (0 for default camera, 1 for external camera)
screen_width, screen_height = 1920, 1080
capture.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", width, height)

smoothed_frame = None  # Initialize smoothed frame
point = None
points = []
state = "measurement"  # State variable: "calibration" or "measurement"

transform_matrix = None  # Initialize the transformation matrix
def nothing(x): pass

cv2.namedWindow("Canny Controls")
cv2.createTrackbar("Low", "Canny Controls", 35, 255, nothing)
cv2.createTrackbar("High", "Canny Controls", 110, 255, nothing)

    # In your loop:
# Mouse callback function to set points during calibration
def draw_circle(event, x, y, flags, param):
    global point, picked_hsv, lower_color, upper_color

    if state == "calibration":
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)

    elif state == "measurement":
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
            picked_hsv = hsv[y, x]
            print(f"Picked HSV: {picked_hsv}")
            lower_color = np.maximum(picked_hsv - hsv_tolerance, [0, 60, 0])  # S >= 60
            upper_color = np.minimum(picked_hsv + hsv_tolerance, [179, 255, 180])  # V <= 180

# Calibration mode function, here 4 points are clicked to calibrate the camera
def calibration_mode():
    global point, points, state, transform_matrix
    # Draw points during calibration
    point_labels = ["Bottom-Left", "Top-Left", "Bottom-Right", "Top-Right"]

    # Display a permanent message in the bottom-left corner
    if len(points) < len(point_labels):
        label = point_labels[len(points)]
        cv2.putText(frame, f"Please calibrate the {label} point", 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
    # Draw all points in the points list as red circles
    for p in points:
        cv2.circle(frame, p, 1, (0, 0, 255), -1)  # Red circle for saved points

    if point is not None or len(points) > 0:
        if len(points) < len(point_labels):
            label = point_labels[len(points)]  # Get the correct label for the current point
            cv2.circle(frame, point, 1, (0, 255, 0), -1)  # Green circle for point

        if len(points) == 4:
            # All points are set display done calibration message
            cv2.putText(frame, "Calibration Points Set", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        if key == ord('n'):
            if len(points) < 4:
                points.append(point)  # Save point
                point = None  # Reset point 
                cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)  # Ensure the window stays on top

           
        if key == ord('r'):
            points = []
            point = None
            cv2.putText(frame, "Points reset", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Camera", frame)
            cv2.waitKey(1000)  # Display the message for 1 second

        if key == ord('c'):
            if len(points) == 4:
                # Define the source points (calibration points in the image)
                src_points = np.array(points, dtype=np.float32)

                # Define the destination points (real-world coordinates)
                dst_points = np.array([left_bottom, left_top, right_bottom, right_top], dtype=np.float32)

                # Compute the perspective transformation matrix
                transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                cv2.putText(frame, "Calibration Finished", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Camera", frame)
                cv2.waitKey(1000)  # Display the message for 1 second
                state = "measurement"
            else:
                cv2.putText(frame, "Please set all 4 points", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Camera", frame)
                cv2.waitKey(1000)  # Display the message for 1 second

    # Draw the information box in the top-right corner
    info_x, info_y = frame.shape[1] - 250, 5  # Top-right corner
    cv2.rectangle(frame, (info_x, info_y), (info_x + 240, info_y + 105), (50, 50, 50), -1)  # Background
    cv2.putText(frame, "Calibration Mode:", (info_x + 10, info_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "Left Click: Calibration Point", (info_x + 10, info_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "'n': Next Calibration Point", (info_x + 10, info_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "'r': Reset points", (info_x + 10, info_y + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "'c': Finish Calibration", (info_x + 10, info_y + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "'ESC': Exit", (info_x + 10, info_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
# Here measurements are taken based on calibration and the picked color of the tentacle area
def measurement_mode():
    global transform_matrix, points, state, point, lower_color, upper_color

    info_x, info_y = frame.shape[1] - 250, 10
    cv2.rectangle(frame, (info_x, info_y), (info_x + 240, info_y + 110), (50, 50, 50), -1)
    cv2.putText(frame, "Measurement Mode:", (info_x + 10, info_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Left Click: Pick Tentacle Color", (info_x + 10, info_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "'r': Reset Calibration", (info_x + 10, info_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "'ESC': Exit", (info_x + 10, info_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Camera Calibrated", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    low = cv2.getTrackbarPos("Low", "Canny Controls")
    high = cv2.getTrackbarPos("High", "Canny Controls")
    edges = cv2.Canny(gray, low, high)

    # Optional cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area_mm2 = 0 

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px > pixel_filter:
            contour = contour.astype(np.float32)  # Ensure float
            contour = contour.reshape(-1, 1, 2)   # Reshape for transform
            try:
                transformed_contour = cv2.perspectiveTransform(contour, transform_matrix)
                real_area = cv2.contourArea(transformed_contour)
                total_area_mm2 += real_area
                cv2.drawContours(frame, [contour.astype(np.int32)], -1, (0, 255, 255), 2)
            except Exception as e:
                print("Transform error:", e)

    # Show area
    cv2.putText(frame, f"Contact Area: {total_area_mm2:.2f} mm^2", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    if key == ord('r'):
        points.clear()
        transform_matrix = None
        point = None
        state = "calibration"



while True:
    ret, frame = capture.read()
    if not ret:
        break

    if smoothed_frame is None:
        smoothed_frame = frame.astype(np.float32)
    else:
        smoothed_frame = alpha * frame.astype(np.float32) + (1 - alpha) * smoothed_frame

    frame_smoothed = smoothed_frame.astype(np.uint8)

    cv2.setMouseCallback("Camera", draw_circle, param=frame_smoothed)


    cv2.setMouseCallback("Camera", draw_circle, param=frame)

    key = cv2.waitKey(1)

    if state == "calibration":
        calibration_mode()
    elif state == "measurement" and transform_matrix is None:
        # Transformation matrix is not set, set it to the identity matrix
        transform_matrix = np.eye(3, dtype=np.float32)
        measurement_mode()
    elif state == "measurement" and transform_matrix is not None:
        # If the transformation matrix is set, use it for measurement
        measurement_mode() 
    
    # Display the frame
    cv2.imshow("Camera", frame)

    if key == 27:  # ESC key to exit
        break

capture.release()
cv2.destroyAllWindows()