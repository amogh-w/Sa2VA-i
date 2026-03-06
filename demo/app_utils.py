import numpy as np
from PIL import Image
import cv2
from debug_utils import debug_info, debug_error, debug_success

markdown_default = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
<style>
        .highlighted-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            color: rgb(255, 255, 239);
            background-color: rgb(225, 231, 254);
            border-radius: 7px;
            padding: 5px 7px;
            display: inline-block;
        }
        .regular-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 400;
            font-size: 14px;
        }
        .highlighted-response {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            border-radius: 6px;
            padding: 3px 4px;
            display: inline-block;
        }
</style>
<span class="highlighted-text" style='color:rgb(107, 100, 239)'>Sa2VA-i</span>
"""

description = """
**Usage** : <br>
&ensp;(1) For **Grounded Caption Generation** Interleaved Segmentation, input prompt like: *"Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer."* <br>
&ensp;(2) For **Segmentation Output**, input prompt like: *"Can you please segment xxx in the given image"* <br>
&ensp;(3) For **Image Captioning** VQA, input prompt like: *"Could you please give me a detailed description of the image?"* <br>
&ensp;(4) For **Image Conversation**, input arbitrary text instruction. <br>
"""

ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0

def desaturate(rgb, factor=0.65):
    """
    Desaturate an RGB color by a given factor.

    :param rgb: A tuple of (r, g, b) where each value is in [0, 255].
    :param factor: The factor by which to reduce the saturation.
                   0 means completely desaturated, 1 means original color.
    :return: A tuple of desaturated (r, g, b) values in [0, 255].
    """
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = rgb_to_hls(r, g, b)
    l = factor
    new_r, new_g, new_b = hls_to_rgb(h, l, s)
    return (int(new_r * 255), int(new_g * 255), int(new_b * 255))

def rgb_to_hls(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    sumc = (maxc+minc)
    rangec = (maxc-minc)
    l = sumc/2.0
    if minc == maxc:
        return 0.0, l, 0.0
    if l <= 0.5:
        s = rangec / sumc
    else:
        s = rangec / (2.0-sumc)
    rc = (maxc-r) / rangec
    gc = (maxc-g) / rangec
    bc = (maxc-b) / rangec
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, l, s

def hls_to_rgb(h, l, s):
    if s == 0.0:
        return l, l, l
    if l <= 0.5:
        m2 = l * (1.0+s)
    else:
        m2 = l+s-(l*s)
    m1 = 2.0*l - m2
    return (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))

def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < ONE_SIXTH:
        return m1 + (m2-m1)*hue*6.0
    if hue < 0.5:
        return m2
    if hue < TWO_THIRD:
        return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
    return m1

def process_markdown(output_str, colors):
    output_str = output_str.replace("\n", "").replace("  ", " ").replace("<s>", "")\
        .replace("<|im_end|>", '').replace("<|end|>", "")
    output_str = output_str.split("ASSISTANT: ")[-1]

    # markdown_out = output_str.replace('[SEG]', '')
    markdown_out = output_str
    markdown_out = markdown_out.replace(
        "<p>", "<span class='highlighted-response' style='background-color:rgb[COLOR]'>"
    )
    markdown_out = markdown_out.replace("</p>", "</span>")

    for color in colors:
        markdown_out = markdown_out.replace("[COLOR]", str(desaturate(tuple(color))), 1)

    markdown_out = f""" 
    {markdown_out}
    """
    markdown_out = markdown_default + "<p><span class='regular-text'>" + markdown_out
    return markdown_out

def show_mask_pred(image, masks):
    masks = [mask[:1] for mask in masks]
    masks = np.concatenate(masks, axis=0)  # (n, h, w)

    selected_colors = []

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255), [255, 192, 203],  # Pink
              [165, 42, 42],    # Brown
              [255, 165, 0],    # Orange
              [128, 0, 128],     # Purple
              [0, 0, 128],       # Navy
              [128, 0, 0],      # Maroon
              [128, 128, 0],    # Olive
              [70, 130, 180],   # Steel Blue
              [173, 216, 230],  # Light Blue
              [255, 192, 0],    # Gold
              [255, 165, 165],  # Light Salmon
              [255, 20, 147],   # Deep Pink
              ]

    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        selected_colors.append(color)
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]


    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    return image, selected_colors

# def show_mask_pred_video(video, masks):
#     ret_video = []
#     selected_colors = []
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
#               (255, 255, 0), (255, 0, 255), (0, 255, 255),
#               (128, 128, 255), [255, 192, 203],  # Pink
#               [165, 42, 42],  # Brown
#               [255, 165, 0],  # Orange
#               [128, 0, 128],  # Purple
#               [0, 0, 128],  # Navy
#               [128, 0, 0],  # Maroon
#               [128, 128, 0],  # Olive
#               [70, 130, 180],  # Steel Blue
#               [173, 216, 230],  # Light Blue
#               [255, 192, 0],  # Gold
#               [255, 165, 165],  # Light Salmon
#               [255, 20, 147],  # Deep Pink
#               ]
#     for i_frame in range(len(video)):
#         frame_masks = [mask[i_frame:i_frame+1] for mask in masks]
#         frame_masks = np.concatenate(frame_masks, axis=0)
#         _mask_image = np.zeros((frame_masks.shape[1], frame_masks.shape[2], 3), dtype=np.uint8)

#         for i, mask in enumerate(frame_masks):
#             if i_frame == 0:
#                 color = colors[i % len(colors)]
#                 selected_colors.append(color)
#             else:
#                 color = selected_colors[i]
#             _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
#             _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
#             _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]

#         # --- FIX START ---
#         # Check if the frame is a file path (string) or an actual image object
#         frame_obj = video[i_frame]
        
#         if isinstance(frame_obj, str) or isinstance(frame_obj, np.str_):
#             # It's a file path, load the image using PIL
#             image = np.array(Image.open(frame_obj).convert("RGB"))
#         else:
#             # It's already an image object/array
#             image = np.array(frame_obj)
            
#         # Ensure image is float for the multiplication below
#         image = image.astype(float)
#         # --- FIX END ---

#         image = image * 0.5 + _mask_image * 0.5
#         image = image.astype(np.uint8)
#         ret_video.append(image)
#     return ret_video, selected_colors

from pathlib import Path
import numpy as np
from PIL import Image

def show_mask_pred_video(video, masks):
    debug_info("Starting video mask overlay...")

    ret_video = []
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 128, 255), (255, 192, 203),
        (165, 42, 42), (255, 165, 0), (128, 0, 128), (0, 0, 128)
    ]
    
    # Determine how many objects we have to select colors once
    num_objs = len(masks)
    selected_colors = [colors[i % len(colors)] for i in range(num_objs)]
    
    debug_info(f"Processing {len(video)} frames for {num_objs} objects.")

    for i_frame in range(len(video)):
        # Load the frame image
        frame_obj = video[i_frame]
        if isinstance(frame_obj, (str, Path)):
            p = Path(frame_obj)
            image = np.array(Image.open(p).convert("RGB"))
        else:
            image = np.array(frame_obj)
        
        orig_h, orig_w = image.shape[:2]

        # Get masks for this specific frame across all objects
        # We use i_frame:i_frame+1 to keep dimensions consistent
        frame_masks = [mask[i_frame:i_frame+1] for mask in masks]
        frame_masks = np.concatenate(frame_masks, axis=0) # (num_objs, h, w)
        
        mask_h, mask_w = frame_masks.shape[1], frame_masks.shape[2]
        
        # Create the mask canvas at model resolution
        _mask_image = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)

        for i, mask in enumerate(frame_masks):
            color = selected_colors[i]
            # Fast boolean indexing instead of manual addition
            _mask_image[mask.astype(bool)] = color

        # Resize mask to original image size if needed
        if (mask_h, mask_w) != (orig_h, orig_w):
            if i_frame == 0:
                debug_info(f"Resizing mismatch detected: Mask({mask_w}x{mask_h}) -> Image({orig_w}x{orig_h})")
            
            _mask_image = np.array(
                Image.fromarray(_mask_image).resize((orig_w, orig_h), resample=Image.NEAREST)
            )

        # Alpha blend
        image = image.astype(np.float32) * 0.5 + _mask_image.astype(np.float32) * 0.5
        ret_video.append(image.astype(np.uint8))
        
        if (i_frame + 1) % 10 == 0:
            debug_info(f"Overlay progress: {i_frame + 1}/{len(video)} frames")

    debug_success("Video overlay complete.")
    return ret_video, selected_colors

def parse_visual_prompts(points):
    ret = {'points': [], 'boxes': []}
    for item in points:
        if item[2] == 1.0:
            ret['points'].append([item[0], item[1]])
        elif item[2] == 2.0 or item[2] == 3.0:
            ret['boxes'].append([item[0], item[1], item[3], item[4]])
        else:
            raise NotImplementedError
    return ret

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

def get_frames_from_video(video_path, n_frames=5, sample_type="uniform"):
    frames = get_video_frames(video_path)
    if sample_type == "uniform":
        stride = len(frames) / (n_frames + 1e-4)
        ret = []
        for i in range(n_frames):
            idx = int(i * stride)
            frame = frames[idx]
            frame = frame[:, :, ::-1]
            frame_image = Image.fromarray(frame).convert('RGB')
            ret.append(frame_image)
    else:
        ret = []
        for frame in frames[:500]:
            frame = frame[:, :, ::-1]
            frame_image = Image.fromarray(frame).convert('RGB')
            ret.append(frame_image)
    return ret

def preprocess_video(video_path, text):
    if "Segment" in text or "segment" in text:
        sample_type = 'begin'
    else:
        sample_type = 'uniform'
    return get_frames_from_video(video_path, sample_type=sample_type)

import os
import cv2

def image2video_and_save(frames, save_path):
    # 1. Ensure the directory exists
    temp_dir = "session_videos_temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        debug_info(f"Created directory: {temp_dir}")

    # 2. If save_path is just a filename, move it into the temp dir
    # If it's already a path, this ensures it stays in the right place
    if not os.path.dirname(save_path):
        save_path = os.path.join(temp_dir, save_path)

    success = frames_to_video(frames, save_path)
    return save_path if success else None


def frames_to_video(frames, output_path: str, fps: int = 24) -> bool:
    if not frames:
        debug_error("No frames provided to frames_to_video")
        return False
        
    try:
        # Convert RGB to BGR for OpenCV
        frames_bgr = [frame[:, :, ::-1] for frame in frames]
        height, width = frames_bgr[0].shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            debug_error(f"Failed to open VideoWriter for path: {output_path}")
            return False

        # Process each frame
        for frame in frames_bgr:
            out.write(frame)

        # Release video writer
        out.release()
        debug_info(f"Video saved successfully to {output_path}")
        return True

    except Exception as e:
        debug_error(f"Error converting frames to video: {str(e)}")
        return False