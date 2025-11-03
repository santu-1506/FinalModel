---
title: CRISPR-BERT Prediction API
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: latest
app_file: app.py
pinned: false
---

# CRISPR-BERT Prediction API

CRISPR off-target prediction API using hybrid CNN-BERT architecture deployed on Hugging Face Spaces.

## 🚀 API Endpoints

- `GET /` - API information
- `GET /health` - Health check (shows if model is loaded)
- `POST /predict` - Make a single prediction
- `POST /batch_predict` - Make batch predictions
- `GET /model/info` - Get model information

## 📝 Usage

### Single Prediction

```bash
curl -X POST https://santu0032-crispr-bert-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"sgRNA": "GGTGAGTGAGTGTGTGCGTGTGG", "DNA": "TGTGAGTGTGTGTGTGTGTGTGT"}'
```

### Response

```json
{
  "prediction": 0,
  "confidence": 0.9919,
  "probabilities": {
    "class_0": 0.9919,
    "class_1": 0.0081
  },
  "threshold_used": 0.65,
  "sgRNA": "GGTGAGTGAGTGTGTGCGTGTGG",
  "DNA": "TGTGAGTGTGTGTGTGTGTGTGT",
  "timestamp": "2025-10-31T..."
}
```

## 📋 Model Requirements

- **sgRNA**: Exactly 23 nucleotides (A, T, C, G, or - for indels)
- **DNA**: Exactly 23 nucleotides (A, T, C, G, or - for indels)

## 🔧 Files Structure

```
.
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── sequence_encoder.py       # Sequence encoding utilities
├── data_loader.py           # Data loading utilities
├── final1/
│   └── weight/
│       ├── final_model.keras           # Trained model
│       ├── threshold_schedule.json    # Threshold config
│       └── bert_weight/                # BERT weights
└── README.md                # This file
```

## 🧬 About CRISPR-BERT

This API uses a hybrid CNN-BERT architecture to predict CRISPR off-target effects:
- **CNN Branch**: Multi-scale convolutions for sequence pattern recognition
- **BERT Branch**: Transformer attention for contextual understanding
- **BiGRU Layers**: Bidirectional GRU for sequence modeling
- **Final Output**: Binary classification (on-target vs off-target)

## 📊 Model Architecture

- **Input**: 23-nt sgRNA and DNA sequences
- **CNN Encoding**: 26x7 one-hot encoding
- **BERT Encoding**: 26 token IDs
- **Output**: Binary prediction with confidence scores

## 🔗 Integration

Update your backend to use this API:

```bash
MODEL_API_URL=https://santu0032-crispr-bert-api.hf.space
```

## 📝 License

MIT License

## 🙏 Credits

Built with TensorFlow, Flask, and deployed on Hugging Face Spaces.
