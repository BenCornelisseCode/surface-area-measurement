import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

video_dir = "/captured_frames/2025-05-20_13-29-44"
sam2_checkpoint = "/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.1])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# take a look the first video frame
torch.cuda.empty_cache()
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# ...existing code above...

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1     # unique id for the object

clicked_points = []
clicked_labels = []

def on_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: ({x}, {y})")
        # Always keep only the latest click
        clicked_points.clear()
        clicked_labels.clear()
        clicked_points.append([x, y])
        clicked_labels.append(1)  # positive click

        points = np.array(clicked_points, dtype=np.float32)
        labels = np.array(clicked_labels, np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        ax = event.inaxes
        ax.clear()
        ax.set_title(f"frame {ann_frame_idx}")
        ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, ax)
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])
        plt.draw()


# Interactive clicking
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title(f"frame {ann_frame_idx}")
ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
input("Press Enter after you are done clicking points and have closed the plot window...")

# After closing the plot, propagate the mask across the video
torch.cuda.empty_cache()
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# ...existing code above...

# --- Refine a specific frame with a negative click interactively ---
ann_frame_idx = 50  # frame to refine
ann_obj_id = 1      # unique id for the object

neg_points = []
neg_labels = []

def on_negative_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Negative click at: ({x}, {y})")
        # Always keep only the latest negative click
        neg_points.clear()
        neg_labels.clear()
        neg_points.append([x, y])
        neg_labels.append(0)  # negative click

        points = np.array(neg_points, dtype=np.float32)
        labels = np.array(neg_labels, np.int32)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        ax = event.inaxes
        ax.clear()
        ax.set_title(f"frame {ann_frame_idx} -- after negative click")
        ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, ax)
        show_mask((out_mask_logits > 0.0).cpu().numpy(), ax, obj_id=ann_obj_id)
        plt.draw()


# Interactive negative click
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title(f"frame {ann_frame_idx} -- click to add negative point")
ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
cid_neg = fig.canvas.mpl_connect('button_press_event', on_negative_click)
plt.show()
input("Press Enter after you are done clicking a negative point and have closed the plot window...")

# After negative click, propagate again
torch.cuda.empty_cache()
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# --- Propagate the refined mask across the video ---
torch.cuda.empty_cache()
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# --- Display the segmented video as an animation ---
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(9, 6))

def update(frame_idx):
    ax.clear()
    ax.set_title(f"frame {frame_idx}")
    img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    ax.imshow(img)
    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            show_mask(out_mask, ax, obj_id=out_obj_id)
    ax.axis('off')


ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(len(frame_names)),
    interval=50,  # ms between frames
    repeat=False
)

plt.show()  # <-- Use this in a script to display the animation window