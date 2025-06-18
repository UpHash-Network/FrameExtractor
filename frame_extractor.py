import os
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from datetime import datetime
import shutil
import tempfile
import time
import threading

# Global variables for processing stop flag and preview images
stop_processing = False
current_preview_images = []

def reset_stop_flag():
    """Reset the processing stop flag"""
    global stop_processing, current_preview_images
    stop_processing = False
    current_preview_images = []

def set_stop_flag():
    """Set the processing stop flag and return preview images"""
    global stop_processing
    stop_processing = True
    return get_preview_images()

def is_video_file(filename):
    """Check if a file is a video file"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    return filename.lower().endswith(video_extensions)

def find_video_files(directory):
    """Recursively search for video files in the specified directory"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_video_file(file):
                video_files.append(os.path.join(root, file))
    return video_files

def save_frame(frame, output_file, output_format, quality):
    """Save frame as image"""
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create temporary file (with delete=False)
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_frame_{os.path.basename(output_file)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save image
        if output_format == "jpg":
            success = cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            success = cv2.imwrite(temp_path, frame)
        
        if not success:
            print(f"Warning: Failed to save to temporary file: {temp_path}")
            return False
        
        # Move temporary file to target location
        try:
            # Wait a bit for file release
            time.sleep(0.1)
            
            # Attempt to move
            shutil.move(temp_path, output_file)
            print(f"Frame saved: {output_file}")
            return True
        except Exception as e:
            print(f"Warning: Failed to move file: {str(e)}")
            # Try to delete temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return False
            
    except Exception as e:
        print(f"Error: Error occurred while saving frame: {str(e)}")
        # Try to delete temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        return False

def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_brightness_contrast(image):
    """Calculate image brightness and contrast"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    return mean_brightness, contrast

def is_good_quality_frame(image, sharpness_threshold=100, brightness_range=(50, 200), contrast_threshold=20):
    """Check if frame has good quality"""
    sharpness = calculate_sharpness(image)
    brightness, contrast = calculate_brightness_contrast(image)
    
    # Sharpness check
    if sharpness < sharpness_threshold:
        return False, f"Low sharpness: {sharpness:.2f}"
    
    # Brightness check
    if not (brightness_range[0] <= brightness <= brightness_range[1]):
        return False, f"Inappropriate brightness: {brightness:.2f}"
    
    # Contrast check
    if contrast < contrast_threshold:
        return False, f"Low contrast: {contrast:.2f}"
    
    return True, f"Good quality: Sharpness={sharpness:.2f}, Brightness={brightness:.2f}, Contrast={contrast:.2f}"

def find_best_frame_in_range(cap, center_frame, search_range=5):
    """Find the best quality frame around the specified frame"""
    best_frame = None
    best_score = -1
    best_frame_index = center_frame
    
    # Set search range
    start_frame = max(0, center_frame - search_range)
    end_frame = center_frame + search_range + 1
    
    original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Quality evaluation
        is_good, reason = is_good_quality_frame(frame)
        if is_good:
            sharpness = calculate_sharpness(frame)
            if sharpness > best_score:
                best_score = sharpness
                best_frame = frame.copy()
                best_frame_index = frame_idx
    
    # Return to original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
    
    return best_frame, best_frame_index, best_score

def evaluate_frame_batch(cap, start_frame, end_frame, n_best=3):
    """Evaluate frames in the specified range and return Best-N"""
    frame_scores = []
    original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_idx in range(start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Quality evaluation
        sharpness = calculate_sharpness(frame)
        brightness, contrast = calculate_brightness_contrast(frame)
        
        # Calculate total score (weighted)
        # Prioritize sharpness, but also consider brightness and contrast
        brightness_score = max(0, 100 - abs(brightness - 125))  # 125 as ideal value
        contrast_score = min(contrast, 100)  # Cap contrast at 100
        
        total_score = (sharpness * 0.7) + (brightness_score * 0.2) + (contrast_score * 0.1)
        
        frame_scores.append({
            'frame_idx': frame_idx,
            'frame': frame.copy(),
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'total_score': total_score
        })
    
    # Return to original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
    
    # Sort by score and return top N
    frame_scores.sort(key=lambda x: x['total_score'], reverse=True)
    return frame_scores[:n_best]

def find_sharpest_frame_in_range(cap, target_frame, search_range, sharpness_threshold):
    """Find the sharpest frame that exceeds the threshold around the specified frame"""
    best_frame = None
    best_sharpness = -1
    best_frame_index = target_frame
    
    # Set search range
    start_frame = max(0, target_frame - search_range)
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), target_frame + search_range + 1)
    
    original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Calculate sharpness
        sharpness = calculate_sharpness(frame)
        
        # Check if it exceeds threshold and is better than current best
        if sharpness >= sharpness_threshold and sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()
            best_frame_index = frame_idx
    
    # Return to original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
    
    return best_frame, best_frame_index, best_sharpness

def extract_frames_from_video(
    video_path,
    frame_interval,
    output_format,
    quality,
    progress_func,
    video_index,
    total_videos,
    enable_quality_check=False,
    sharpness_threshold=100,
    search_range=5
):
    """Extract frames from a video file"""
    global stop_processing, current_preview_images
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], f"Error: Could not open video file: {video_path}"
    
    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
    
    # Delete existing output directory and recreate
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    extracted_count = 0
    skipped_count = 0
    saved_frames = []
    
    # Process frames
    for frame_num in range(0, total_frames, frame_interval):
        if stop_processing:
            break
        
        # Update progress
        progress_desc = f"[{video_index + 1}/{total_videos}] Processing {video_name} - Frame {frame_num}/{total_frames}"
        overall_progress = (video_index / total_videos) + ((frame_num / total_frames) / total_videos)
        progress_func(overall_progress, desc=progress_desc)
        
        if enable_quality_check:
            # Find sharpest frame in range
            best_frame, best_frame_index, best_sharpness = find_sharpest_frame_in_range(
                cap, frame_num, search_range, sharpness_threshold
            )
            
            if best_frame is not None:
                # Save the best frame
                output_filename = f"frame_{extracted_count + 1:06d}_{best_frame_index:06d}.{output_format}"
                output_file = os.path.join(output_path, output_filename)
                
                if save_frame(best_frame, output_file, output_format, quality):
                    extracted_count += 1
                    if len(saved_frames) < 3:  # Save first 3 frames for preview
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        saved_frames.append(pil_image)
                        current_preview_images.append(pil_image)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
                print(f"Frame {frame_num} skipped: No frame exceeding sharpness threshold {sharpness_threshold}")
        else:
            # Extract frame at specified interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                output_filename = f"frame_{extracted_count + 1:06d}_{frame_num:06d}.{output_format}"
                output_file = os.path.join(output_path, output_filename)
                
                if save_frame(frame, output_file, output_format, quality):
                    extracted_count += 1
                    if len(saved_frames) < 3:  # Save first 3 frames for preview
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        saved_frames.append(pil_image)
                        current_preview_images.append(pil_image)
                else:
                    skipped_count += 1
    
    cap.release()
    
    # Create result text
    result_text = f"=== {video_name} ===\n"
    result_text += f"    Total frames: {total_frames} ({fps:.2f} FPS)\n"
    result_text += f"    Extracted frames: {extracted_count}\n"
    if enable_quality_check:
        result_text += f"    Skipped frames: {skipped_count}\n"
        result_text += f"    Sharpness threshold: {sharpness_threshold}\n"
        result_text += f"    Search range: ±{search_range} frames\n"
    result_text += f"    Output directory: {output_path}"
    
    return saved_frames, result_text

def process_directory(
    input_dir,
    frame_interval,
    output_format,
    quality,
    enable_quality_check,
    sharpness_threshold,
    search_range,
    progress=gr.Progress()
):
    """Process videos in directory"""
    global stop_processing, current_preview_images
    reset_stop_flag()  # Reset flag at start of processing
    
    if not os.path.exists(input_dir):
        return "Error: Specified directory not found."
    
    print(f"Processing started: {input_dir}")
    
    # Initial progress display
    progress(0, desc="Searching for video files...")
    
    video_files = find_video_files(input_dir)
    if not video_files:
        return "Error: No video files found in the specified directory."
    
    print(f"Found video files: {len(video_files)}")
    for video in video_files:
        print(f"- {video}")
    
    all_results = []
    total_videos = len(video_files)
    
    # Process each video file
    for video_index, video_path in enumerate(video_files):
        if stop_processing:
            final_result = "\n\n".join(all_results) + "\n\nProcessing was stopped."
            return final_result
        
        # Display current video processing status
        previews, result = extract_frames_from_video(
            video_path,
            frame_interval,
            output_format,
            quality,
            progress,
            video_index,
            total_videos,
            enable_quality_check,
            sharpness_threshold,
            search_range
        )
        
        if previews:
            current_preview_images.extend(previews)
        all_results.append(result)
        
        if stop_processing:
            break
    
    # Completion message
    progress(1.0, desc="All processing completed!")
    
    return "\n\n".join(all_results)  # Return text only

def get_preview_images():
    """Get preview images when stopped or completed"""
    global current_preview_images
    return current_preview_images[:3]  # Return only first 3 images

def on_process_complete():
    """Update preview images when processing is complete"""
    return get_preview_images()

# Create Gradio interface
with gr.Blocks(title="Video Frame Extractor") as demo:
    gr.Markdown("# Video Frame Extractor")
    gr.Markdown("Extract frames from multiple video files and save them in individual folders.")
    
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(
                label="Input Directory",
                placeholder="Enter the path of the directory containing videos to process",
                interactive=True
            )
            
            with gr.Row():
                frame_interval = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=30,
                    step=1,
                    label="Frame Interval (extract every N frames)",
                    info="The optimal frame will be selected from around frames at this interval"
                )
                quality = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=95,
                    step=1,
                    label="JPEG Quality (for jpg format)"
                )
            
            output_format = gr.Radio(
                choices=["jpg", "png"],
                value="png",
                label="Output Format"
            )
            
            # Quality check feature settings
            with gr.Accordion("Sharpness-Based Selection Settings", open=True):
                enable_quality_check = gr.Checkbox(
                    label="Enable Sharpness-Based Selection",
                    value=True,
                    info="Select the sharpest frame that exceeds the threshold from around frames at specified intervals"
                )
                
                with gr.Row():
                    sharpness_threshold = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Sharpness Threshold",
                        info="Only select frames with sharpness above this value"
                    )
                    search_range = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Search Range",
                        info="How many frames before and after the specified frame to search"
                    )
                
                gr.Markdown("""
                **Sharpness-Based Selection Operation:**
                1. Use frames at specified intervals (e.g., every 30 frames) as reference points
                2. Evaluate all frames within ±search range around each reference point
                3. Select the frame with highest sharpness that exceeds the threshold
                4. Skip if no frame exceeds the threshold
                
                **When disabled:** Save frames at specified intervals as-is
                """)
            
            with gr.Row():
                extract_button = gr.Button("Start Frame Extraction", variant="primary")
                stop_button = gr.Button("Stop Processing", variant="stop")
        
        with gr.Column():
            result_text = gr.Textbox(
                label="Processing Results", 
                lines=10,
                max_lines=15,
                show_copy_button=True
            )
            preview_gallery = gr.Gallery(
                label="Preview (First 3 frames)",
                show_label=True,
                columns=3,
                height=300,
                object_fit="cover"
            )
    
    # Event handlers
    extract_button.click(
        fn=process_directory,
        inputs=[
            input_dir,
            frame_interval,
            output_format,
            quality,
            enable_quality_check,
            sharpness_threshold,
            search_range
        ],
        outputs=[result_text]
    ).then(
        fn=on_process_complete,
        inputs=[],
        outputs=[preview_gallery]
    )
    
    stop_button.click(
        fn=set_stop_flag,
        inputs=[],
        outputs=[preview_gallery]
    )

if __name__ == "__main__":
    demo.launch() 