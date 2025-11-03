#!/usr/bin/env python3
"""
CRISPR-BERT Prediction API - Production Version
Rebuilds model architecture and loads weights
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime

# Import required modules
logger.info("Importing model definition modules...")
try:
    from sequence_encoder import encode_for_cnn, encode_for_bert
    from final1.train_model import build_crispr_bert_model
    logger.info("✓ Successfully imported model modules")
except ImportError as e:
    logger.error(f"✗ Failed to import modules: {e}")
    build_crispr_bert_model = None

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
threshold = 0.7  # Default from training
model_loaded = False

# Paths
WEIGHTS_PATH = "final1/weight/final_model_compatible.h5"
THRESHOLD_PATH = "final1/weight/threshold_schedule.json"


def load_model():
    """
    Load the CRISPR-BERT model by rebuilding architecture and loading weights
    """
    global model, threshold, model_loaded
    
    try:
        logger.info("=" * 70)
        logger.info("LOADING CRISPR-BERT MODEL")
        logger.info("=" * 70)
        
        # Check if build function is available
        if build_crispr_bert_model is None:
            logger.error("❌ build_crispr_bert_model function not available")
            return False
        
        # Check if weights file exists
        if not os.path.exists(WEIGHTS_PATH):
            logger.error(f"❌ Weights file not found: {WEIGHTS_PATH}")
            logger.info(f"Current directory: {os.getcwd()}")
            if os.path.exists('final1/weight'):
                logger.info(f"Files in final1/weight: {os.listdir('final1/weight')}")
            return False
        
        logger.info(f"✓ Found weights file: {WEIGHTS_PATH}")
        file_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        logger.info(f"✓ Weights size: {file_size_mb:.2f} MB")
        
        # Step 1: Build model architecture
        logger.info("\nStep 1: Building model architecture...")
        model = build_crispr_bert_model()
        logger.info("✓ Model architecture created")
        
        # Step 2: Load weights
        logger.info("\nStep 2: Loading weights...")
        model.load_weights(WEIGHTS_PATH)
        logger.info("✓ Weights loaded successfully")
        
        # Step 3: Compile model for inference
        logger.info("\nStep 3: Compiling model...")
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("✓ Model compiled")
        
        # Step 4: Load threshold
        if os.path.exists(THRESHOLD_PATH):
            logger.info(f"\nStep 4: Loading threshold from {THRESHOLD_PATH}")
            with open(THRESHOLD_PATH, 'r') as f:
                data = json.load(f)
                threshold = float(data.get('final_threshold', 0.7))
            logger.info(f"✓ Using threshold: {threshold:.3f}")
        else:
            logger.warning(f"Threshold file not found. Using default: {threshold}")
        
        model_loaded = True
        logger.info("\n" + "=" * 70)
        logger.info("✅ MODEL READY FOR PREDICTIONS")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Failed to load model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def predict_single(sgrna, dna):
    """
    Make prediction for a single sgRNA-DNA pair
    
    Args:
        sgrna: Guide RNA sequence
        dna: Target DNA sequence
    
    Returns:
        dict: Prediction results
    """
    global model, threshold
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Encode sequences
        cnn_input = encode_for_cnn(sgrna, dna)
        token_ids = encode_for_bert(sgrna, dna)
        segment_ids = np.zeros(26, dtype=np.int32)
        position_ids = np.arange(26, dtype=np.int32)
        
        # Add batch dimension
        inputs = {
            'cnn_input': cnn_input[np.newaxis, ...],
            'token_ids': token_ids[np.newaxis, ...],
            'segment_ids': segment_ids[np.newaxis, ...],
            'position_ids': position_ids[np.newaxis, ...]
        }
        
        # Make prediction
        probabilities = model.predict(inputs, verbose=0)
        
        # Apply threshold
        predicted_class = int(probabilities[0, 1] >= threshold)
        confidence = float(probabilities[0, predicted_class])
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'class_0': float(probabilities[0, 0]),
                'class_1': float(probabilities[0, 1])
            },
            'threshold_used': float(threshold),
            'class_label': 'cleavage' if predicted_class == 1 else 'no_cleavage'
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information"""
    return jsonify({
        'service': 'CRISPR-BERT Prediction API',
        'status': 'running',
        'model_loaded': model_loaded,
        'version': '2.0.0',
        'model_type': 'Hybrid CNN-BERT-BiGRU',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'batch_predict': '/batch_predict (POST)',
            'model_info': '/model/info'
        },
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.utcnow().isoformat()
    }), 200 if model_loaded else 503


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        return jsonify({
            'model_loaded': True,
            'threshold': float(threshold),
            'architecture': 'CRISPR-BERT (CNN + BERT + BiGRU)',
            'components': {
                'cnn': 'Inception-based CNN for sequence features',
                'bert': 'BERT-like transformer for context',
                'bigru': 'Bidirectional GRU for sequence modeling'
            },
            'input_format': {
                'sgRNA': 'RNA sequence (20-23 nucleotides)',
                'DNA': 'DNA sequence (matching sgRNA length)'
            },
            'output_format': {
                'prediction': '0 (no cleavage) or 1 (cleavage)',
                'confidence': 'Probability of predicted class',
                'probabilities': {'class_0': 'no cleavage prob', 'class_1': 'cleavage prob'},
                'class_label': 'Human-readable label'
            },
            'performance': {
                'validation_accuracy': '82.5%',
                'f1_score': 0.72,
                'auroc': 0.97
            }
        })
    except Exception as e:
        logger.error(f"Error in model_info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a single sgRNA-DNA pair"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait and try again.',
            'status': 'loading'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Accept both cases
        sgrna = data.get('sgRNA') or data.get('sgrna')
        dna = data.get('DNA') or data.get('dna')
        
        if not sgrna or not dna:
            return jsonify({
                'error': 'Missing required fields',
                'required': ['sgRNA', 'DNA']
            }), 400
        
        # Validate sequences
        if len(sgrna) != len(dna):
            return jsonify({
                'error': 'sgRNA and DNA must have the same length'
            }), 400
        
        # Make prediction
        result = predict_single(sgrna, dna)
        result['sgRNA'] = sgrna
        result['DNA'] = dna
        result['timestamp'] = datetime.utcnow().isoformat()
        
        logger.info(f"Prediction: {sgrna[:10]}... -> {result['prediction']} (conf: {result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple sgRNA-DNA pairs"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait and try again.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'sequences' not in data:
            return jsonify({'error': 'No sequences provided'}), 400
        
        sequences = data['sequences']
        if not isinstance(sequences, list):
            return jsonify({'error': 'sequences must be a list'}), 400
        
        results = []
        for idx, seq in enumerate(sequences):
            sgrna = seq.get('sgRNA') or seq.get('sgrna')
            dna = seq.get('DNA') or seq.get('dna')
            
            if not sgrna or not dna:
                results.append({
                    'index': idx,
                    'error': 'Missing sgRNA or DNA'
                })
                continue
            
            try:
                result = predict_single(sgrna, dna)
                result['index'] = idx
                result['sgRNA'] = sgrna
                result['DNA'] = dna
                results.append(result)
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e),
                    'sgRNA': sgrna,
                    'DNA': dna
                })
        
        logger.info(f"Batch prediction: {len(results)} sequences processed")
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===================================================================
# APPLICATION STARTUP
# ===================================================================

if __name__ == '__main__':
    # Development mode - load model synchronously
    logger.info("=" * 70)
    logger.info("RUNNING IN DEVELOPMENT MODE")
    logger.info("=" * 70)
    
    load_model()
    
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"\nStarting Flask server on port {port}...")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
else:
    # Production mode (gunicorn) - load model in background
    logger.info("=" * 70)
    logger.info("RUNNING IN PRODUCTION MODE (GUNICORN)")
    logger.info("=" * 70)
    
    import threading
    import time
    
    def load_model_background():
        """Load model in background thread"""
        logger.info("\n[Background Thread] Waiting 3 seconds for server to start...")
        time.sleep(3)
        
        logger.info("[Background Thread] Starting model loading...")
        logger.info(f"[Background Thread] Current directory: {os.getcwd()}")
        logger.info(f"[Background Thread] Weights path: {WEIGHTS_PATH}")
        logger.info(f"[Background Thread] Weights exist: {os.path.exists(WEIGHTS_PATH)}")
        
        try:
            success = load_model()
            if success:
                logger.info("\n" + "=" * 70)
                logger.info("✓✓✓ MODEL LOADED SUCCESSFULLY IN BACKGROUND ✓✓✓")
                logger.info("=" * 70)
            else:
                logger.error("\n" + "=" * 70)
                logger.error("✗✗✗ MODEL LOADING FAILED ✗✗✗")
                logger.error("=" * 70)
        except Exception as e:
            logger.error(f"\n[Background Thread] Exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Start background loading thread
    model_thread = threading.Thread(
        target=load_model_background,
        daemon=False,
        name="ModelLoader"
    )
    model_thread.start()
    logger.info("✓ Background model loading thread started")
    logger.info("=" * 70)

