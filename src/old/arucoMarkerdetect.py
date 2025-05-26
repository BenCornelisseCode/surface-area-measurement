import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    cap = cv2.VideoCapture(4)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    screen_width, screen_height = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            # Stack all corners to compute the global centroid
            all_corners = np.concatenate([c[0] for c in corners], axis=0)
            global_center = all_corners.mean(axis=0)
            global_center_int = tuple(global_center.astype(int))
            cv2.circle(frame, global_center_int, 8, (255, 0, 255), 2)  # Draw global center

            for i, corner in enumerate(corners):
                c = corner[0]
                # Find the corner furthest from the global centroid
                dists = np.linalg.norm(c - global_center, axis=1)
                furthest_idx = np.argmax(dists)
                furthest_corner = c[furthest_idx]
                furthest_corner_int = tuple(furthest_corner.astype(int))
                # Draw a red circle at the furthest corner
                cv2.circle(frame, furthest_corner_int, 10, (0, 0, 255), 3)
                # Draw a line from global center to furthest corner
                cv2.line(frame, global_center_int, furthest_corner_int, (0, 0, 255), 2)
                # Optionally, draw marker ID
                center = c.mean(axis=0)
                center_int = tuple(center.astype(int))
                cv2.putText(frame, f"ID:{ids[i][0]}", center_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow('Aruco Marker Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()