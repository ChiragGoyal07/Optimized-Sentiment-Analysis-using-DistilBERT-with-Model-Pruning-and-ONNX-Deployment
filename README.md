# Optimized Sentiment Analysis using DistilBERT with Model Pruning and ONNX Deployment

## Project Overview

This repository demonstrates a practical machine-learning pipeline:

- **Model Architecture**: Hugging Face’s DistilBERT, a lightweight Transformer.
- **Training & Optimization**: Uses IMDb movie reviews dataset to train a sentiment classifier.
- **Model Pruning**: Applies unstructured magnitude-based pruning using PyTorch APIs.
- **Quantization**: Applies post-training dynamic quantization.
- **Deployment**: Exports the model to ONNX format and performs inference with ONNX Runtime.
- **Benchmarking**: Compares accuracy, inference speed, and model size before/after pruning and quantization.

## Tech Stack

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [PyTorch Pruning APIs](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Datasets](https://github.com/huggingface/datasets)

## Setup Instructions

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/optimized-distilbert-sentiment-analysis.git
   cd optimized-distilbert-sentiment-analysis
   ```
