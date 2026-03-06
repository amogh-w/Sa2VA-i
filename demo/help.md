# Running the application

```bash
python demo.py \
    --model_id "kumuji/Sa2VA-i-1B" \
    --host "127.0.0.1" \
    --port 7860 \
    --max_frames 300 \
    --max_width 1920 \
    --max_height 1080 \
    --example_dir "examples" \
    --temp_dir "session_masks_temp"
```

## Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--model_id` | The Hugging Face model ID to load and use for inference. | `"kumuji/Sa2VA-i-1B"`, `"kumuji/Sa2VA-i-4B"`, `"kumuji/Sa2VA-i-8B"`, `"kumuji/Sa2VA-i-26B"` |
| `--host` | The IP address or hostname where the Gradio server will be hosted. | `"127.0.0.1"` |
| `--port` | The port number on which the Gradio server will listen. | `7860` |
| `--max_frames` | The maximum number of frames for video. | `300` |
| `--max_width` | The maximum width (in pixels) for video. | `1920` |
| `--max_height` | The maximum height (in pixels) for video. | `1080` |
| `--example_dir` | The directory path containing preset example images or videos for the UI. | `"examples"` |
| `--temp_dir` | The directory where temporary session masks and files will be stored. | `"session_masks_temp"` |