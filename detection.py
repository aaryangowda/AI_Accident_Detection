import tensorflow as tf
import numpy as np
import cv2
import gc
import os
import logging

logger = logging.getLogger(__name__)

class AccidentDetectionModel:
    def __init__(self, model_json_path, model_weights_path):
        try:
            # Configure TensorFlow for CPU
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            
            # Verify files exist and have content
            if not os.path.exists(model_json_path):
                raise FileNotFoundError(f"Model JSON file not found: {model_json_path}")
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")
                
            # Check file sizes
            json_size = os.path.getsize(model_json_path)
            weights_size = os.path.getsize(model_weights_path)
            if json_size < 1000:  # JSON should be at least 1KB
                raise ValueError(f"Model JSON file too small: {json_size} bytes")
            if weights_size < 1000000:  # Weights should be at least 1MB
                raise ValueError(f"Model weights file too small: {weights_size} bytes")
            
            # Load model architecture
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Create model from JSON
            self.model = tf.keras.models.model_from_json(loaded_model_json)
            
            # Load weights directly (no chunking needed as we verified the file)
            self.model.load_weights(model_weights_path)
            
            # Convert to TFLite with dynamic range quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
            # Simple representative dataset
            def representative_dataset():
                for _ in range(10):
                    data = np.random.rand(1, 250, 250, 3) * 255
                    yield [data.astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            
            # Convert model
            logger.info("Converting model to TFLite format...")
            tflite_model = converter.convert()
            
            # Create interpreter
            self.interpreter = tf.lite.Interpreter(
                model_content=tflite_model,
                num_threads=2
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()
            
            # Get details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Clean up
            del self.model
            del tflite_model
            del converter
            tf.keras.backend.clear_session()
            gc.collect()
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def predict_accident(self, frame):
        try:
            # Ensure minimal memory usage during prediction
            input_data = frame.astype(np.int8)
            
            # Set tensor and run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            pred = output_data[0]
            
            # Clean prediction memory
            del output_data
            gc.collect()
            
            return "Accident" if pred[0] > 0 else "No Accident", pred
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Error", np.array([[0.0]])
    
    def __del__(self):
        try:
            del self.interpreter
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass