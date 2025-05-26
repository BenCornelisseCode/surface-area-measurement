import cv2
import numpy as np
import pandas as pd
import torch
import cv2.aruco as aruco
import matplotlib.animation as animation
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

class FrameSource:
    def __init__(self, video_dir):
        self.frame_files = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ], key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        self.idx = 0

    def read_frame(self, index):
        if 0 <= index < len(self.frame_files):
            return cv2.imread(self.frame_files[index])
        else:
            raise IndexError("Frame index out of range.")


    def release(self):
        pass

class SAM2Model:
    def __init__(self, video_dir, sam2_checkpoint, model_cfg, device=None):
        self.video_dir = video_dir
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        print(f"using device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        torch.cuda.empty_cache()
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.predictor.reset_state(self.inference_state)
        self.ann_obj_id = 1

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.1])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.1])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def interactive_annotate(self, frame_idx=0):
        clicked_points = []
        clicked_labels = []

        def on_click(event):
            if event.inaxes:
                x, y = int(event.xdata), int(event.ydata)
                print(f"Clicked at: ({x}, {y})")
                clicked_points.clear()
                clicked_labels.clear()
                clicked_points.append([x, y])
                clicked_labels.append(1)
                points = np.array(clicked_points, dtype=np.float32)
                labels = np.array(clicked_labels, np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=self.ann_obj_id,
                    points=points,
                    labels=labels,
                )
                ax = event.inaxes
                ax.clear()
                ax.set_title(f"frame {frame_idx}")
                ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
                self.show_points(points, labels, ax)
                self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])
                plt.draw()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"frame {frame_idx}")
        ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        input("Press Enter after you are done clicking points and have closed the plot window...")

    def propagate(self):
        torch.cuda.empty_cache()
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def refine_negative(self, frame_idx):
        neg_points = []
        neg_labels = []

        def on_negative_click(event):
            if event.inaxes:
                x, y = int(event.xdata), int(event.ydata)
                print(f"Negative click at: ({x}, {y})")
                neg_points.clear()
                neg_labels.clear()
                neg_points.append([x, y])
                neg_labels.append(0)
                points = np.array(neg_points, dtype=np.float32)
                labels = np.array(neg_labels, np.int32)
                _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=self.ann_obj_id,
                    points=points,
                    labels=labels,
                )
                ax = event.inaxes
                ax.clear()
                ax.set_title(f"frame {frame_idx} -- after negative click")
                ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
                self.show_points(points, labels, ax)
                self.show_mask((out_mask_logits > 0.0).cpu().numpy(), ax, obj_id=self.ann_obj_id)
                plt.draw()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"frame {frame_idx} -- click to add negative point")
        ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
        cid_neg = fig.canvas.mpl_connect('button_press_event', on_negative_click)
        plt.show()
        input("Press Enter after you are done clicking a negative point and have closed the plot window...")

    def animate(self, video_segments):
        fig, ax = plt.subplots(figsize=(9, 6))

        def update(frame_idx):
            ax.clear()
            ax.set_title(f"frame {frame_idx}")
            img = Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            ax.imshow(img)
            if frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
            ax.axis('off')

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(len(self.frame_names)),
            interval=50,
            repeat=False
        )
        plt.show()

class VideoSegmentationPipeline:
    def __init__(self, frame_source, segmentation_model):
        self.frame_source = FrameSource(frame_source)
        self.segmentation_model = segmentation_model
        self.transform_matrix = None
        self.calibrated = False
        self.masks = {}
        self.frame_times = []
        self.frame_names = []
        self.video_writer = None

        # Define destination points for the perspective transformation in mm
        self.dst_points = np.array([
            [0, 0],  # Bottom-left  
            [0, 180],  # Top-left
            [380, 0],  # Bottom-right
            [380, 180] # Top-right
        ], dtype=np.float32)

    def display_information(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Display the calibration status on the frame
        if self.calibrated:
            print("Calibration successful.")
        else:
            print("Calibration not yet done. Finding Markers.")

    def undistort_camera_frame(self, frame, calibration_params):
        # Load camera calibration parameters
        calibration_data = np.load(calibration_params)
        camera_matrix = calibration_data["camera_matrix"]
        dist_coeffs = calibration_data["dist_coefficients"]
        width = int(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(frame.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        return undistorted_frame

    def calibrate_area(self, frame):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # Only proceed if 4 markers are detected
        if ids is not None and len(ids) == 4:

            # Find the center of the markers
            all_corners = np.concatenate([c[0] for c in corners], axis=0)
            global_center = all_corners.mean(axis=0)

            # Find the furthest corner from the global center
            marker_points = []
            for c in corners:
                dists = np.linalg.norm(c[0] - global_center, axis=1)
                furthest_idx = np.argmax(dists)
                furthest_corner = c[0][furthest_idx]
                marker_points.append(tuple(furthest_corner))

            points_np = np.array(marker_points)
            # Sort the points first in y and then in x direction
            # following the order: bottom-left, top-left, top-right, bottom-right
            print(points_np, "points_np before sorting")
            idx = np.lexsort((points_np[:,1], points_np[:,0]))
            sort = points_np[idx]
            src_points = np.array([sort[0], sort[1], sort[2], sort[3]], dtype=np.float32)
        else:
            print("Not enough markers detected for calibration.")
            return

        if len(src_points) == 4:
            # Define the destination points for the transformation

            # Compute the transformation matrix
            self.transform_matrix = cv2.getPerspectiveTransform(src_points, self.dst_points)
            self.calibrated = True

    def calculate_area_from_mask(self, frame, mask):
        # Calculate the area of the mask
        area = np.sum(mask)
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Take the top-right corner of dst_points as the A4 paper size
        a4_width_mm, a4_height_mm = self.dst_points[2][0], self.dst_points[1][1]  # Bottom-right corner x, Top-left corner y
        warped_size = (int(a4_width_mm), int(a4_height_mm))  # (width, height) in mm

        warped_mask = cv2.warpPerspective(mask, self.transform_matrix, warped_size)
        warped_area = np.sum(warped_mask > 0)  # Count non-zero pixels
        return warped_area

    def load_model(self, frame, frame_idx):
        # Use SAM2 to segment, store mask in self.masks[frame_idx]
        pass

    def save_results(self, excel_path, video_out_path):
        # Prepare data storage
        records = []
        mask_arrays = {}

        # Prepare video writer
        first_frame = self.frame_source.read_frame(0)
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out_path, fourcc, 20.0, (width, height))

        for frame_idx, mask_dict in self.masks.items():
            frame = self.frame_source.read_frame(frame_idx)
            if frame is None:
                continue

            for obj_id, mask in mask_dict.items():
                # Resize mask to frame size if needed
                mask_resized = cv2.resize(mask.astype(np.uint8)*255, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Visualize mask overlay
                overlay = frame.copy()
                overlay[mask_resized > 0] = [0, 0, 255]  # Red overlay for mask
                vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                # Write frame to video
                out.write(vis)

                # Calculate area and center
                area = np.sum(mask_resized > 0)
                ys, xs = np.where(mask_resized > 0)
                if len(xs) > 0 and len(ys) > 0:
                    center_x = int(np.mean(xs))
                    center_y = int(np.mean(ys))
                else:
                    center_x, center_y = -1, -1

                # Save record
                records.append({
                    'frame_idx': frame_idx,
                    'object_id': obj_id,
                    'area': area,
                    'center_x': center_x,
                    'center_y': center_y
                })

                # Save mask array for this frame/object
                mask_arrays[f"frame{frame_idx}_obj{obj_id}"] = mask_resized

        out.release()

        # Save Excel file
        df = pd.DataFrame(records)
        df.to_excel(excel_path, index=False)

        # Optionally, save all masks as a compressed numpy file for later use
        np.savez_compressed(excel_path.replace('.xlsx', '_masks.npz'), **mask_arrays)

        print(f"Results saved to {excel_path} and {video_out_path}")

    def run(self):

        # Main loop to read frames, calibrate, segment, and save results
        frame_idx = 0

        while True:
            frame = self.frame_source.read_frame(frame_idx)
            if frame is None:
                break
            # Undistort the camera frame
            self.calibrate_area(frame)

            if not self.calibrated:
                self.display_information(frame)
                frame_idx += 1
            elif frame_idx >= len(self.frame_source.frame_files):
                print("End of video frames reached, no calibration found")
                break
            else:
                self.display_information(frame)
                print(f"Found calibration markers in frame {frame_idx}, proceeding with segmentation...")
                continue

        print("Calibration complete, proceeding with segmentation...")
        frame_idx = 0

        self.segmentation_model.interactive_annotate(frame_idx)
        video_segments = self.segmentation_model.propagate()
        self.segmentation_model.refine_negative(frame_idx)
        self.segmentation_model.animate(video_segments)
        self.masks = video_segments

        # Calculate areas from masks
        for frame_idx, mask_dict in video_segments.items():
            for obj_id, mask in mask_dict.items():
                area = self.calculate_area_from_mask(self.frame_source.read_frame(frame_idx), mask)
                print(f"Frame {frame_idx}, Object ID {obj_id}: Area = {area} pixels")



        self.save_results("results.xlsx", "segmented_output.mp4")


if __name__ == "__main__":
    # Remove this line - don't change working directory
    # os.chdir('/home/ben/surface-area-measurement/sam2')
    
    # Define the configuration with absolute paths
    video_dir = "/home/ben/surface-area-measurement/captured_frames/2025-05-20_13-29-44"
    sam2_checkpoint = "/home/ben/surface-area-measurement/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "/home/ben/surface-area-measurement/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    
    sam2_model = SAM2Model(video_dir, sam2_checkpoint, model_cfg)
    pipeline = VideoSegmentationPipeline(video_dir, sam2_model)
    pipeline.run()