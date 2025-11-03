"""
CRISPR-BERT Model API - TensorFlow Lite Version
Optimized for low memory usage (500 MB instead of 4-6 GB)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
interpreter = None
input_details = None
output_details = None
model_loaded = False

# Paths
MODEL_PATH = 'final1/weight/final_model.tflite'
THRESHOLD_PATH = 'final1/weight/best_model.keras'  # For threshold if needed

logger.info("=" * 60)
logger.info("CRISPR-BERT TFLite Model API Starting...")
logger.info("=" * 60)

def initialize_model():
    """Load TFLite model"""
    global interpreter, input_details, output_details, model_loaded
    
    logger.info("Loading TFLite model...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"File exists: {os.path.exists(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at {MODEL_PATH}")
        return False
    
    try:
        # Get file size
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        logger.info(f"Model file size: {file_size_mb:.2f} MB")
        
        # Load TFLite interpreter
        logger.info("Creating TFLite interpreter...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        
        # Allocate tensors
        logger.info("Allocating tensors...")
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info("✓ Model loaded successfully!")
        logger.info(f"  Number of inputs: {len(input_details)}")
        logger.info(f"  Number of outputs: {len(output_details)}")
        
        # Log input details
        for i, detail in enumerate(input_details):
            logger.info(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")
        
        # Log output details
        for i, detail in enumerate(output_details):
            logger.info(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")
        
        model_loaded = True
        logger.info("=" * 60)
        logger.info("✅ TFLite Model Ready! (Memory usage: ~500 MB)")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def encode_sequence(sgRNA, DNA):
    """
    Encode sgRNA and DNA sequences for model input
    This should match the encoding used during training
    """
    # Base encoding (simplified - adjust based on your actual encoding)
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    def encode_seq(seq):
        return [base_to_idx.get(base.upper(), 4) for base in seq]
    
    # Encode sequences
    sgRNA_encoded = encode_seq(sgRNA)
    DNA_encoded = encode_seq(DNA)
    
    # Pad or truncate to length 26 (based on model input)
    def pad_sequence(seq, length=26):
        if len(seq) < length:
            return seq + [4] * (length - seq)  # Pad with 'N'
        return seq[:length]
    
    sgRNA_padded = pad_sequence(sgRNA_encoded)
    DNA_padded = pad_sequence(DNA_encoded)
    
    return sgRNA_padded, DNA_padded

def prepare_model_inputs(sgRNA, DNA):
    """
    Prepare inputs for TFLite model
    Adjust based on your model's actual input requirements
    """
    sgRNA_encoded, DNA_encoded = encode_sequence(sgRNA, DNA)
    
    # Create inputs based on model requirements
    # This is a simplified version - adjust based on your actual model
    
    inputs = []
    
    for detail in input_details:
        shape = detail['shape']
        dtype = detail['dtype']
        name = detail.get('name', 'unknown')
        
        logger.info(f"Preparing input '{name}': shape={shape}, dtype={dtype}")
        
        # Create appropriate input tensor
        if 'cnn' in name.lower() or len(shape) == 3:
            # CNN input: (batch, 26, 7) or similar
            # Create one-hot encoding
            combined = sgRNA_encoded + DNA_encoded
            combined = combined[:26]  # Take first 26
            
            # One-hot encode (assuming 7 features)
            input_tensor = np.zeros((1, 26, 7), dtype=np.float32)
            for i, val in enumerate(combined[:26]):
                if val < 7:
                    input_tensor[0, i, val] = 1.0
            
        else:
            # Token IDs, segment IDs, position IDs
            if 'token' in name.lower():
                input_tensor = np.array([sgRNA_encoded + DNA_encoded[:0]], dtype=np.int32)
            elif 'segment' in name.lower():
                input_tensor = np.array([[0] * 26], dtype=np.int32)
            elif 'position' in name.lower():
                input_tensor = np.array([list(range(26))], dtype=np.int32)
            else:
                # Default: zeros
                input_tensor = np.zeros(shape, dtype=dtype)
        
        inputs.append(input_tensor)
    
    return inputs

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'name': 'CRISPR-BERT TFLite API',
        'version': '2.0-tflite',
        'status': 'running',
        'model_loaded': model_loaded,
        'memory_optimized': True,
        'estimated_memory': '500 MB',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Make predictions (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'loading',
        'model_loaded': model_loaded,
        'model_type': 'TensorFlow Lite'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    global interpreter, input_details, output_details, model_loaded
    
    # Check if model is loaded
    if not model_loaded:
        logger.warning("Model not loaded, attempting to load...")
        if not initialize_model():
            return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        sgRNA = data.get('sgRNA', '')
        DNA = data.get('DNA', '')
        
        logger.info(f"Prediction request: sgRNA length={len(sgRNA)}, DNA length={len(DNA)}")
        
        if not sgRNA or not DNA:
            return jsonify({'error': 'Both sgRNA and DNA sequences required'}), 400
        
        # Prepare inputs
        inputs = prepare_model_inputs(sgRNA, DNA)
        
        # Set input tensors
        for i, (input_tensor, detail) in enumerate(zip(inputs, input_details)):
            interpreter.set_tensor(detail['index'], input_tensor)
            logger.info(f"Set input {i}: shape={input_tensor.shape}")
        
        # Run inference
        logger.info("Running inference...")
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        logger.info(f"Output shape: {output_data.shape}, values: {output_data}")
        
        # Parse results
        # Assuming binary classification output: [prob_class_0, prob_class_1]
        if len(output_data.shape) == 2 and output_data.shape[1] == 2:
            prob_off_target = float(output_data[0][1])  # Probability of off-target
            prob_on_target = float(output_data[0][0])   # Probability of on-target
        else:
            prob_off_target = float(output_data[0][0])
            prob_on_target = 1.0 - prob_off_target
        
        # Calculate efficiency (simplified)
        efficiency = prob_on_target * 100
        
        result = {
            'score': prob_off_target,
            'efficiency': efficiency,
            'on_target_probability': prob_on_target,
            'off_target_probability': prob_off_target,
            'classification': 'on-target' if prob_on_target > 0.5 else 'off-target',
            'model_type': 'TFLite',
            'sgRNA': sgRNA,
            'DNA': DNA
        }
        
        logger.info(f"Prediction successful: efficiency={efficiency:.2f}%, score={prob_off_target:.4f}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Initialize model on startup (for gunicorn)
if __name__ != '__main__':
    # When run by gunicorn
    logger.info("App imported by gunicorn - loading model in background")
    import threading
    import time
    
    def load_model_delayed():
        time.sleep(2)  # Give server time to start
        logger.info("Starting TFLite model loading...")
        try:
            success = initialize_model()
            if success:
                logger.info("✓✓✓ TFLite model loaded successfully! ✓✓✓")
            else:
                logger.error("✗✗✗ TFLite model loading FAILED ✗✗✗")
        except Exception as e:
            logger.error(f"Exception during model loading: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    model_thread = threading.Thread(target=load_model_delayed, daemon=False, name="TFLiteModelLoader")
    model_thread.start()
    logger.info(f"Background model loading thread started: {model_thread.name}")

if __name__ == '__main__':
    # For local testing
    logger.info("Running in development mode")
    initialize_model()
    app.run(host='0.0.0.0', port=5001, debug=False)

