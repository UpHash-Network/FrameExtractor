# Video Frame Extractor

This tool is a GUI application for batch extracting frames from multiple video files in a specified directory and saving them as image files.

## Features

- **Batch Processing of Multiple Videos**: Recursively search and batch process video files in specified directories
- **Sharpness-Based Selection**: Automatically select the sharpest and highest quality frames from around specified intervals
- **Frame Extraction Interval Setting**: Specify how many frames to extract (1-100)
- **Quality Control**: Set sharpness threshold and search range to select optimal frames
- **Output Format Selection**: Choose from JPG/PNG formats
- **JPEG Quality Adjustment**: Set compression quality for JPG format (1-100)
- **Processing Stop Function**: Stop processing at any time during execution
- **Progress Display**: Real-time display of processing status
- **Preview Display**: Preview the first 3 extracted frames

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

**Note:** The versions listed in `requirements.txt` are from the author's execution environment. Other versions may work as well, so please adjust versions as needed if dependency errors occur.

## Usage

1. Launch the application:
```bash
python frame_extractor.py
```

2. Configure the following settings in the GUI interface displayed in your browser:

### Basic Settings
   - **Input Directory**: Enter the path of the directory containing video files to process
   - **Frame Interval**: Set how many frames to extract (1-100)
   - **Output Format**: Choose from JPG/PNG formats
   - **JPEG Quality**: Set compression quality for JPG format (1-100)

### Sharpness-Based Selection Settings
   - **Enable Sharpness-Based Selection**: Toggle ON/OFF for high-quality frame automatic selection
   - **Sharpness Threshold**: Only select frames with sharpness above this value (50-500)
   - **Search Range**: How many frames before and after the specified frame to search (1-20)

3. Click the "Start Frame Extraction" button to begin processing.

4. During processing, you can interrupt with the "Stop Processing" button.

5. When processing is complete, a "video_name_frames" folder is created for each video file, containing the extracted frame images.

## About Sharpness-Based Selection

The main feature of this tool is not simply extracting frames at specified intervals, but automatically selecting the sharpest and highest quality frames from around specified frame positions.

### How It Works
1. Use frames at specified intervals (e.g., every 30 frames) as reference points
2. Evaluate all frames within ±search range around each reference point
3. Select the frame with highest sharpness that exceeds the threshold
4. Skip if no frame exceeds the threshold

### Benefits
- Extract only sharp frames with minimal blur
- Automatically exclude frames that are too dark or too bright
- Efficiently obtain high-quality frames from large amounts of video

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)

## Notes

- A "video_name_frames" folder is automatically created for each video file
- Existing output folders are deleted and recreated when processing begins
- Preview shows the first 3 frames during processing
- When sharpness-based selection is enabled, frames not exceeding the threshold are skipped

## Limitations

This is a simplified version designed for general use cases and ease of use. For more advanced video frame processing with professional-grade features, we recommend using **Reflct's SharpeFrames**.

## License

This project is released under the MIT License. 