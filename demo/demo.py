import argparse, os, torch, subprocess, cv2, numpy as np, shutil, zipfile
import gradio as gr
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from app_utils import (
    process_markdown, preprocess_video, show_mask_pred_video, 
    show_mask_pred, image2video_and_save, description
)
from debug_utils import debug_info, debug_error, debug_success

# ==========================================
# 1. Argument Parsing & Global Setup
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sa2VA-i Gradio Demo")
    
    # Model Config
    parser.add_argument("--model_id", type=str, default="kumuji/Sa2VA-i-1B", help="HF Model ID to load")
    
    # Server Config
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the gradio server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the gradio server on")
    
    # Constraints & Constants
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames allowed for video upload")
    parser.add_argument("--max_width", type=int, default=1920, help="Max width allowed for video")
    parser.add_argument("--max_height", type=int, default=1080, help="Max height allowed for video")
    parser.add_argument("--example_dir", type=str, default="examples", help="Directory for preset examples")
    parser.add_argument("--temp_dir", type=str, default="session_masks_temp", help="Directory for temporary mask storage")
    
    return parser.parse_args()

args = parse_args()

model = None
tok = None

def load_model():
    debug_info("Loading model")
    global model, tok
    if model is None:
        debug_info(f"Initializing model from {args.model_id}")
        model = AutoModel.from_pretrained(
            args.model_id, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).eval().cuda()
        tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    return model, tok

# ==========================================
# 2. Utilities
# ==========================================

def convert_to_h264(path):
    debug_info("Converting video to H.264 format to be visible in browsers")
    if not path or not os.path.exists(path): return path
    out = path.replace(".mp4", "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p", out], capture_output=True)
    return out

def reset_all():
    debug_info("Initiating global reset...")
    if os.path.exists(args.temp_dir):
        try:
            shutil.rmtree(args.temp_dir)
            debug_info(f"Deleted temporary directory: {args.temp_dir}")
        except Exception as e:
            debug_error(f"Cleanup error: {e}")

    debug_success("All states and UI components cleared.")
    
    return {
        media_path_state: None,
        media_type_state: "none",
        selected_state: [],
        all_masks_state: [],
        file_input: None,
        timeline: gr.update(value=[], visible=False),
        video_input_row: gr.update(visible=False),
        video_output_row: gr.update(visible=False),
        image_row: gr.update(visible=False),
        chat_col: gr.update(visible=False),
        visual_output_header: gr.update(visible=False),
        selection_tray: gr.update(value=[], visible=False),
        out_gallery: gr.update(value=[], visible=False),
        out_video: gr.update(value=None, visible=False),
        chat: [],
        msg: "",
        mask_file: None,
        reset_row: gr.update(visible=False),
        input_image_display: None,
        output_image_display: None
    }

def handle_upload(files):
    debug_info("Handling file upload request")
    
    def upload_reset():
        debug_info("Returning safe upload reset state")
        return {
            media_path_state: None, timeline: gr.update(value=[], visible=False), 
            selection_tray: gr.update(value=[], visible=False), out_video: gr.update(value=None, visible=False),
            media_type_state: "none", video_input_row: gr.update(visible=False), 
            video_output_row: gr.update(visible=False), image_row: gr.update(visible=False), 
            input_image_display: None, output_image_display: None,
            out_gallery: gr.update(value=[], visible=False), chat_col: gr.update(visible=False), 
            visual_output_header: gr.update(visible=False), reset_row: gr.update(visible=False)
        }

    if not files:
        debug_error("No files provided by user.")
        return upload_reset()
    
    is_multi = isinstance(files, list) and len(files) > 1
    common_updates = {
        chat_col: gr.update(visible=True),
        visual_output_header: gr.update(visible=True),
        reset_row: gr.update(visible=True)
    }

    # CASE 1: MULTIPLE IMAGES
    if is_multi:
        debug_info(f"Processing {len(files)} uploaded images")
        try:
            frames = []
            paths = []
            for i, f in enumerate(files):
                frames.append((Image.open(f.name).convert("RGB"), f"Image {i}"))
                paths.append(f.name)
                
            debug_success("Multi-image upload processed.")
            return {
                **common_updates,
                media_path_state: paths,
                media_type_state: "multi-image",
                timeline: gr.update(value=frames, visible=True),
                video_input_row: gr.update(visible=True),
                selection_tray: gr.update(value=[], visible=True),
                image_row: gr.update(visible=False),
                input_image_display: None,
                output_image_display: None,
                out_gallery: gr.update(visible=True, value=[]),
                video_output_row: gr.update(visible=False),
                out_video: gr.update(visible=False)
            }
        except Exception as e:
            debug_error("Multi-image processing failed:", e)
            return upload_reset()

    # Single file logic
    file = files[0] if isinstance(files, list) else files
    is_video = file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    # CASE 2: VIDEO
    if is_video:
        cap = cv2.VideoCapture(file.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > args.max_frames:
            cap.release()
            raise gr.Error(f"Video too long ({total_frames} frames). Max is {args.max_frames}.")

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > args.max_width or height > args.max_height:
            cap.release()
            raise gr.Error(f"Resolution too high ({width}x{height}). Max: {args.max_width}x{args.max_height}.")

        frames, count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append((Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), f"{count}"))
            count += 1
        cap.release()
        
        debug_success(f"Video decoded: {count} frames total.")

        return {
            **common_updates,
            media_path_state: file.name,
            media_type_state: "video",
            timeline: gr.update(value=frames, visible=True),
            video_input_row: gr.update(visible=True),
            selection_tray: gr.update(value=[], visible=True),
            out_video: gr.update(visible=True, value=None),
            video_output_row: gr.update(visible=True),
            image_row: gr.update(visible=False),
            input_image_display: None,
            output_image_display: None,
            out_gallery: gr.update(visible=True, value=[])
        }
    
    # CASE 3: SINGLE IMAGE
    else:
        debug_info(f"Processing single image: {file.name}")
        try:
            img = Image.open(file.name).convert("RGB")
            debug_success("Image upload processed.")
            return {
                **common_updates,
                media_path_state: file.name,
                media_type_state: "image",
                timeline: gr.update(value=[], visible=False),
                video_input_row: gr.update(visible=False),
                selection_tray: gr.update(visible=False),
                out_video: gr.update(visible=False),
                video_output_row: gr.update(visible=False),
                image_row: gr.update(visible=True),
                input_image_display: img,
                output_image_display: None,
                out_gallery: gr.update(visible=False)
            }
        except Exception as e:
            debug_error("Single image processing failed:", e)
            return upload_reset()
        
def save_session_masks_zip(all_masks_history):
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir, exist_ok=True)

    if not all_masks_history: return None

    session_dir = os.path.join(args.temp_dir, f"session_{os.getpid()}")
    if os.path.exists(session_dir): shutil.rmtree(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    
    zip_path = os.path.join(session_dir, "session_masks.zip")
    total_saved = 0

    debug_info(f"Preparing ZIP in: {session_dir}")

    try:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for turn_idx, turn_masks in enumerate(all_masks_history):
                for obj_idx, m in enumerate(turn_masks):
                    # Convert to numpy array immediately to handle shapes easily
                    if hasattr(m, "cpu"):
                        m = m.cpu().numpy()
                    
                    m = np.squeeze(m)
                    
                    # CASE A: Standard 2D Image Mask (H, W)
                    if m.ndim == 2:
                        mask_list = [m]
                    # CASE B: Video Mask Volume (Frames, H, W)
                    elif m.ndim == 3:
                        debug_info(f"Expanding mask volume for Turn {turn_idx+1}, Obj {obj_idx+1}: {m.shape}")
                        mask_list = [m[i] for i in range(m.shape[0])]
                    else:
                        debug_error(f"Unsupported mask shape: {m.shape}")
                        continue

                    # Process the frames in the list
                    for f_idx, frame in enumerate(mask_list):
                        fname = f"turn_{turn_idx+1}_obj_{obj_idx+1}_f{f_idx}.png"
                        p = os.path.join(session_dir, fname)
                        
                        # Save and Zip
                        Image.fromarray((frame * 255).astype('uint8')).convert('L').save(p)
                        zf.write(p, fname)
                        os.remove(p)
                        total_saved += 1

        debug_success(f"Successfully zipped {total_saved} masks to {zip_path}")
        return zip_path

    except Exception as e:
        debug_error(f"Critical error during ZIP creation: {str(e)}")
        return None

# ==========================================
# 3. Processing Logic
# ==========================================

def process(v, msg, chat_history, selected_info, draft_mode, total_masks_history, media_type):
    """Main controller for processing all media types."""
    if not v or not msg:
        debug_error("Process triggered but missing video/image or message.")
        return chat_history, gr.skip(), gr.skip(), gr.skip(), None, total_masks_history

    debug_info(f"Processing started. Media Type: {media_type} | Draft Mode: {draft_mode}")
    m_instance, t_instance = load_model()

    if media_type == "image":
        return _handle_image_logic(v, msg, chat_history, m_instance, t_instance, total_masks_history)
    else:
        return _handle_temporal_logic(v, msg, chat_history, selected_info, draft_mode, 
                                     total_masks_history, media_type, m_instance, t_instance)

# --- INTERNAL WORKER FUNCTIONS ---

def _handle_image_logic(v, msg, chat_history, m_instance, t_instance, total_masks_history):
    debug_info("Executing Image Logic Path")
    img = Image.open(v).convert("RGB")
    
    # Build history string
    past_text = "".join([f"User: {u}\nAssistant: {a}\n" for u, a in chat_history])

    input_dict = {
        "image": img, "text": f"<image>{msg}", "past_text": past_text,
        "tokenizer": t_instance, "return_drafts_only": False
    }

    with torch.inference_mode():
        out = m_instance.predict_forward(**input_dict)
    
    clean_resp = out["prediction"].split("<|im_end|>")[0].strip()
    masks = out.get("prediction_masks", [])

    if not masks:
        debug_info("No masks predicted for image.")
        chat_history.append((msg, clean_resp))
        return chat_history, gr.skip(), gr.skip(), gr.skip(), None, total_masks_history

    res_image, colors = show_mask_pred(img, masks)
    formatted_resp = process_markdown(clean_resp, colors)
    
    chat_history.append((msg, formatted_resp))
    total_masks_history.append(masks)
    
    zip_out = save_session_masks_zip(total_masks_history)
    debug_success("Image processing complete.")
    return chat_history, res_image, gr.skip(), gr.skip(), zip_out, total_masks_history


def _handle_temporal_logic(v, msg, chat_history, selected_info, draft_mode, 
                          total_masks_history, media_type, m_instance, t_instance):
    debug_info(f"Executing Temporal Logic Path ({media_type})")
    original_images = []
    
    # 1. Gather Frames
    if media_type == "multi-image":
        original_images = [np.array(Image.open(img_path).convert("RGB")) for img_path in v]
        if selected_info:
            sample_idx = sorted([int(info[1].split()[-1]) for info in selected_info])
        else:
            sample_idx = list(range(len(original_images)))
    else:
        # Video extraction
        original_images = _extract_video_frames(v)
        sample_idx = sorted([int(info[1]) for info in selected_info]) if selected_info else [0]
        
    debug_info(f"Processing {len(original_images)} total frames, sampling {len(sample_idx)} frames.")

    # 2. Model Inference
    model_text = f"<image>{msg.replace('<image>', '')}"
    model_input_images = [Image.fromarray(img) for img in original_images]

    with torch.inference_mode():
        out = m_instance.predict_forward(
            video=model_input_images, 
            sample_idx=sample_idx, 
            text=model_text, 
            tokenizer=t_instance, 
            return_drafts_only=draft_mode 
        )
    
    clean_resp = out["prediction"].split("<|im_end|>")[0].strip()
    masks = out.get("prediction_masks", [])

    if not masks:
        debug_info("No masks predicted for temporal input.")
        chat_history.append((msg, clean_resp))
        return chat_history, gr.skip(), gr.skip(), gr.skip(), None, total_masks_history

    total_masks_history.append(masks)
    zip_out = save_session_masks_zip(total_masks_history)
    
    # 3. Visualization logic
    bg_source = [original_images[i] for i in sample_idx] if (draft_mode or media_type == "multi-image") else original_images
    display_frames, colors = show_mask_pred_video(bg_source, masks)
    formatted_resp = process_markdown(clean_resp, colors)
    chat_history.append((msg, formatted_resp))

    if draft_mode or media_type == "multi-image":
        debug_success("Returning gallery output.")
        return chat_history, gr.skip(), display_frames, gr.skip(), zip_out, total_masks_history
    else:
        debug_info("Compiling final video output...")
        raw_path = image2video_and_save(display_frames, f"out_{os.getpid()}.mp4")
        final_path = convert_to_h264(raw_path)
        debug_success(f"Video compiled: {final_path}")
        return chat_history, gr.skip(), gr.skip(), final_path, zip_out, total_masks_history


def _extract_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# ==========================================
# 4. Gradio UI Layout
# ==========================================

def auto_sample_uniform(timeline_data):
    """Picks 12 frames evenly across the uploaded timeline/video."""
    if not timeline_data or len(timeline_data) == 0:
        debug_error("Uniform sampling failed: No frames in timeline.")
        return [], []
    
    total_frames = len(timeline_data)
    # Generate 12 indices (handles cases with < 12 frames gracefully)
    num_samples = min(12, total_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
    
    new_selection = []
    for idx in indices:
        item = timeline_data[idx]
        # In Gradio Gallery value, item[0] is the path/dict, item[1] is the caption
        if isinstance(item[0], dict):
            path = item[0]['path']
        else:
            path = item[0]
        caption = item[1]
        new_selection.append((path, caption))
    
    debug_success(f"Uniformly sampled {num_samples} frames from total {total_frames}.")
    return new_selection, new_selection

def select_example(evt: gr.SelectData):
    """Safely extracts the file path from a gallery selection event."""
    try:
        # Get selected index
        idx = evt.index
        
        # Handle both list and tuple formats
        selected_item = PRESET_EXAMPLES[idx]
        
        # Extract path - handle different formats
        if isinstance(selected_item, (list, tuple)):
            path = selected_item[0]  # First element is path
        elif isinstance(selected_item, dict):
            path = selected_item.get('path') or selected_item.get('name')
        else:
            path = selected_item
        
        # Create a file-like object for Gradio
        import tempfile
        
        if os.path.exists(path):
            # Create a temporary file with the same content
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1])
            
            # Copy the file
            import shutil
            shutil.copy2(path, temp_file.name)
            temp_file.close()
            
            debug_success(f"Example selected: {path} -> {temp_file.name}")
            return [temp_file.name]
        
        debug_error(f"File not found: {path}")
        return None
        
    except Exception as e:
        debug_error(f"Example selection error: {e}")
        return None

# Example Data
PRESET_EXAMPLES = []
if os.path.exists(args.example_dir):
    for f in os.listdir(args.example_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.mov')):
            PRESET_EXAMPLES.append((os.path.join(args.example_dir, f), f))

with gr.Blocks(theme=gr.themes.Soft(), title="Sa2VA-i Demo") as demo:
    selected_state = gr.State([])
    all_masks_state = gr.State([])
    media_path_state = gr.State(None)
    media_type_state = gr.State("none")

    gr.Markdown("# Sa2VA-i Demo")
    gr.Markdown(description)
    
    # Examples Gallery
    if len(PRESET_EXAMPLES) == 0:
        gr.Markdown("Please add media to examples folder")
    else:
        with gr.Accordion("Pick from an Example", open=False):
            example_gallery = gr.Gallery(
                value=PRESET_EXAMPLES, 
                label="Preset Examples", 
                columns=8, object_fit="contain", height="80px"
            )

    file_input = gr.File(label="Upload Image(s) or Video", file_count="multiple")

    with gr.Row(visible=False) as video_input_row:
        timeline = gr.Gallery(label="Timeline / Image Set", columns=10, rows=2, object_fit="contain", allow_preview=False)

    with gr.Row():
        with gr.Column(scale=2):
            visual_output_header = gr.Markdown("### Visual Output", visible=False)
            
            with gr.Column() as video_tools:
                selection_tray = gr.Gallery(label="Selection Tray", columns=8, object_fit="contain", height="120px", visible=False)
                with gr.Row():
                    btn_clear_selection = gr.Button("Clear Selection", size="sm", variant="secondary", visible=False)
                    btn_auto_sample = gr.Button("Auto-Sample 12 Frames", size="sm", variant="secondary", visible=False)
            
            with gr.Row(visible=False) as image_row:
                input_image_display = gr.Image(label="Original Image", type="pil", interactive=False)
                output_image_display = gr.Image(label="Segmented Result", type="pil", interactive=False)
            
            out_gallery = gr.Gallery(label="Output Results", columns=2, object_fit="contain", height="500px", visible=False)
        
        with gr.Column(scale=1, visible=False) as chat_col:
            gr.Markdown("### Analysis & Chat")
            chat = gr.Chatbot(label="Chat History", height=450)
            msg = gr.Textbox(label="Instruction", placeholder="e.g., 'segment the red car'")
            btn_run = gr.Button("Chat", variant="primary")
            mask_file = gr.File(label="Download Masks (.zip)")

    with gr.Column(visible=False) as video_output_row:
        btn_final = gr.Button("Generate Final Video", variant="primary")
        out_video = gr.Video(label="Final Video Result")

    with gr.Row(visible=False) as reset_row:
        btn_reset = gr.Button("Reset Everything", variant="stop")

    # --- LISTENERS ---

    # This feeds the path into file_input, which then triggers handle_upload
    if len(PRESET_EXAMPLES) > 0:
        example_gallery.select(
            fn=select_example,
            inputs=None,
            outputs=file_input
        )

    file_input.change(
        fn=handle_upload, 
        inputs=file_input, 
        outputs=[
            media_path_state, timeline, selection_tray, out_video, 
            media_type_state, video_input_row, video_output_row, 
            image_row, input_image_display, output_image_display, 
            out_gallery, chat_col, visual_output_header, reset_row
        ]
    ).then(fn=lambda: debug_info("File load complete."))

    btn_reset.click(
        fn=reset_all,
        inputs=None,
        outputs=[
            media_path_state, media_type_state, selected_state, all_masks_state,
            file_input, timeline, video_input_row, video_output_row,
            image_row, chat_col, visual_output_header, selection_tray,
            out_gallery, out_video, chat, msg, mask_file, reset_row,
            input_image_display, output_image_display 
        ]
    )

    media_type_state.change(
        fn=lambda x: [gr.update(visible=(x in ["video", "multi-image"]))] * 2, 
        inputs=media_type_state, 
        outputs=[btn_clear_selection, btn_auto_sample]
    )

    def on_select(evt: gr.SelectData, current):
        debug_info(f"UI Selection: '{evt.value['caption']}' at {evt.value['image']['path']}")
        current.append((evt.value['image']['path'], evt.value['caption']))
        return current, current
    
    timeline.select(on_select, selected_state, [selected_state, selection_tray])
    
    btn_clear_selection.click(
        fn=lambda: ([], []), 
        inputs=None, 
        outputs=[selected_state, selection_tray]
    ).then(fn=lambda: debug_info("Selection tray cleared by user."))

    btn_auto_sample.click(
        fn=auto_sample_uniform,
        inputs=[timeline],
        outputs=[selected_state, selection_tray]
    )

    btn_run.click(
        fn=lambda: debug_info("Processing trigger: Chat Button clicked (Draft Mode)"),
        inputs=None, outputs=None
    ).then(
        fn=process, 
        inputs=[media_path_state, msg, chat, selected_state, gr.State(True), all_masks_state, media_type_state], 
        outputs=[chat, output_image_display, out_gallery, out_video, mask_file, all_masks_state]
    )
    
    msg.submit(
        fn=lambda: debug_info("Processing trigger: Enter pressed in Textbox (Draft Mode)"),
        inputs=None, outputs=None
    ).then(
        fn=process, 
        inputs=[media_path_state, msg, chat, selected_state, gr.State(True), all_masks_state, media_type_state], 
        outputs=[chat, output_image_display, out_gallery, out_video, mask_file, all_masks_state]
    )

    btn_final.click(
        fn=lambda: debug_info("Processing trigger: Generate Final Video clicked (Full Mode)"),
        inputs=None, outputs=None
    ).then(
        fn=process, 
        inputs=[media_path_state, msg, chat, selected_state, gr.State(False), all_masks_state, media_type_state], 
        outputs=[chat, output_image_display, out_gallery, out_video, mask_file, all_masks_state]
    )

if __name__ == "__main__":
    load_model()
    demo.launch(server_name=args.host, server_port=args.port, share=False)