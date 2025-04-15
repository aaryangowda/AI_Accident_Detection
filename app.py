from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import tensorflow as tf
import time
import gc

# Configure TensorFlow for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Basic thread configuration
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

# Configure TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

from detection import AccidentDetectionModel
from dotenv import load_dotenv
import io
import uvicorn
import gdown
import logging
import shutil
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Model file paths
MODEL_JSON = "model.json"
MODEL_WEIGHTS = "model_weights.h5"

# Google Drive file IDs
MODEL_JSON_ID = os.getenv("MODEL_JSON_ID", "1rTNqBBjEE9XnuWM8FFInOI_1o3Skw7xa")
MODEL_WEIGHTS_ID = os.getenv("MODEL_WEIGHTS_ID", "18dLdwQiubd0yqnNpkM5Pi6G6EM0PxXKg")

# Video file path
VIDEO_PATH = "static/video.mp4"
UPLOAD_DIR = "static/uploads"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables
model = None
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = None
current_video_path = None
frame_skip = 1  # Process every frame for smoothest video
frame_count = 0
MAX_FRAME_SIZE = (640, 480)  # High quality resolution
JPEG_QUALITY = 95  # Maximum JPEG quality
MAX_FPS = 30  # Full frame rate for smooth video

def initialize_model():
    """Initialize the model"""
    global model
    try:
        if model is None:
            model = AccidentDetectionModel(MODEL_JSON, MODEL_WEIGHTS)
            logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def download_model_files():
    """Download model files from Google Drive if they don't exist or are corrupted"""
    try:
        json_url = f"https://drive.google.com/uc?id={MODEL_JSON_ID}"
        weights_url = f"https://drive.google.com/uc?id={MODEL_WEIGHTS_ID}"
        
        if not MODEL_JSON_ID or not MODEL_WEIGHTS_ID:
            raise ValueError("MODEL_JSON_ID and MODEL_WEIGHTS_ID must be set in .env file")
        
        # Clean up existing files if they exist
        if os.path.exists(MODEL_JSON):
            os.remove(MODEL_JSON)
            logger.info("Removed existing model.json")
        
        if os.path.exists(MODEL_WEIGHTS):
            os.remove(MODEL_WEIGHTS)
            logger.info("Removed existing model_weights.h5")
            
        # Download model.json
        logger.info("Downloading model.json...")
        success = gdown.download(url=json_url, output=MODEL_JSON, quiet=False, fuzzy=True)
        if not success or not os.path.exists(MODEL_JSON):
            raise Exception("Failed to download model.json")
        
        # Download model_weights.h5
        logger.info("Downloading model_weights.h5...")
        success = gdown.download(url=weights_url, output=MODEL_WEIGHTS, quiet=False, fuzzy=True)
        if not success or not os.path.exists(MODEL_WEIGHTS):
            raise Exception("Failed to download model_weights.h5")
        
        # Verify file sizes
        weights_size = os.path.getsize(MODEL_WEIGHTS)
        if weights_size < 1000000:  # Less than 1MB is probably wrong
            os.remove(MODEL_WEIGHTS)
            raise Exception(f"Downloaded weights file is too small ({weights_size} bytes)")
            
        logger.info(f"Model files downloaded successfully. Weights size: {weights_size} bytes")
        
        # Initialize model after downloading
        initialize_model()
        
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        # Clean up any partially downloaded files
        if os.path.exists(MODEL_JSON):
            os.remove(MODEL_JSON)
        if os.path.exists(MODEL_WEIGHTS):
            os.remove(MODEL_WEIGHTS)
        raise

# Download model files on startup
download_model_files()

def process_frame(frame):
    """Process a single frame with high quality settings"""
    try:
        # Maintain high quality resolution
        frame = cv2.resize(frame, MAX_FRAME_SIZE)
        
        # Convert to RGB for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        
        # Make prediction
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        
        # Draw prediction with high quality overlay
        if pred == "Accident":
            prob = round(prob[0][0]*100, 2)
            # Larger, more visible overlay
            cv2.rectangle(frame, (0, 0), (120, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Accident", (2, 15), font, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"{prob}%", (2, 35), font, 0.6, (255, 255, 0), 2)
        
        # Convert to JPEG with maximum quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buffer.tobytes()
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def generate_frames():
    global frame_count
    last_frame_time = 0
    
    while True:
        try:
            # Smooth frame rate control
            current_time = time.time()
            if current_time - last_frame_time < 1.0/MAX_FPS:
                continue
            last_frame_time = current_time
            
            ret, frame = video_capture.read()
            if not ret:
                logger.info("End of video reached, restarting...")
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            frame_bytes = process_frame(frame)
            if frame_bytes is None:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error generating frame: {str(e)}")
            break

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global video_capture, current_video_path
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return JSONResponse(
                status_code=400,
                content={"error": "Only MP4, AVI, and MOV files are allowed"}
            )
        
        # Create a unique filename
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update current video path
        current_video_path = file_path
        
        # Release existing video capture if any
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        
        logger.info(f"Video uploaded successfully: {file_path}")
        return JSONResponse(content={"message": "Video uploaded successfully"})
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error uploading video: {str(e)}"}
        )

@app.get("/video_feed")
async def video_feed():
    global video_capture, current_video_path
    
    try:
        if video_capture is None:
            logger.info("Initializing video capture...")
            
            # Try to use uploaded video if available
            if current_video_path and os.path.exists(current_video_path):
                video_source = current_video_path
            else:
                # Try webcam first
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    logger.info("Webcam not available, trying default video file...")
                    video_source = VIDEO_PATH
                else:
                    video_source = 0
            
            video_capture = cv2.VideoCapture(video_source)
            if not video_capture.isOpened():
                error_msg = "Could not open any video source"
                logger.error(error_msg)
                return Response(content=error_msg, status_code=500)
            logger.info("Video capture initialized successfully")
        
        return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        error_msg = f"Error in video feed: {str(e)}"
        logger.error(error_msg)
        return Response(content=error_msg, status_code=500)

@app.on_event("shutdown")
async def shutdown_event():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        logger.info("Video capture released")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True) 