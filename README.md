# AI Accident Detection System

A real-time accident detection system using deep learning and computer vision. This system processes video streams to detect accidents and provide immediate alerts. Built with TensorFlow and FastAPI for efficient and reliable accident detection. Developed by Aaryan Gowda.

## Features

- Real-time accident detection from video streams
- Support for webcam and video file input
- Web interface for easy interaction
- High-quality video processing (640x480 resolution)
- TensorFlow Lite optimization for efficient inference
- Support for multiple video formats (MP4, AVI, MOV)
- Real-time accident probability display
- Optimized for CPU performance
- Low latency video processing

## Tech Stack

- Python 3.8+
- TensorFlow 2.15.0 with TFLite optimization
- OpenCV 4.9.0 for video processing
- FastAPI for backend API
- Bootstrap 5.3 for responsive UI
- WebSocket for real-time streaming

## Setup

1. Clone the repository:
```bash
git clone https://github.com/aaryangowda/AI_Accident_Detection.git
cd AI_Accident_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
MODEL_JSON_ID=your_google_drive_json_id
MODEL_WEIGHTS_ID=your_google_drive_weights_id
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:8080`

## Project Structure

- `app.py`: Main FastAPI application with video processing
- `detection.py`: AI model implementation with TFLite optimization
- `templates/`: HTML templates for web interface
- `static/`: Static files and uploaded videos
- `requirements.txt`: Project dependencies with version specifications

## Contact

For any queries or collaborations, please contact:
- Email: aaryangowda006@gmail.com
- GitHub: [@aaryangowda](https://github.com/aaryangowda)
