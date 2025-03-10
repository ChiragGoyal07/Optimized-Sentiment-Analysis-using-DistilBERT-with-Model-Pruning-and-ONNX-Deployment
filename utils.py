import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import time
import numpy as np

from transformers import (
    Trainer, 
    TrainingArguments, 
    DistilBertForSequenceClassification, 
    DistilBertTokenizerFast
)

import onnx
import onnxruntime

from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(eval_pred):
    """
    Compute metrics for Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}


def prune_model(model, amount=0.2):
    """
    Apply unstructured magnitude-based pruning to linear layers.
    `amount` is the fraction of parameters to prune.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Perform global unstructured pruning across all linear layers
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def remove_pruning_reparam(model):
    """
    After pruning, remove pruning parametrizations to make 
    the model "permanently" pruned.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    return model


def measure_inference_time_pytorch(model, tokenizer, text_list, device='cpu'):
    """
    Measures average inference time on PyTorch model.
    """
    model.eval()
    model.to(device)

    start_time = time.time()

    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors='pt').to(device)
            outputs = model(**inputs)

    end_time = time.time()

    avg_inference_time = (end_time - start_time) / len(text_list)
    return avg_inference_time


def dynamic_quantize_model(model):
    """
    Apply dynamic quantization to reduce model size and potentially speed up inference.
    (works for CPU inference only).
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear},  # Quantize only Linear layers
        dtype=torch.qint8
    )
    return quantized_model


def export_to_onnx(model, tokenizer, output_path='distilbert_pruned_quantized.onnx', max_length=128, device='cpu'):
    """
    Exports the given model to ONNX format.
    """
    model.eval()
    model.to(device)

    # Example input for tracing
    text = "This is a sample input for ONNX export."
    inputs = tokenizer(
        text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length'
    ).to(device)

    # Export
    torch.onnx.export(
        model,                                # model being run
        (inputs["input_ids"], inputs["attention_mask"]),  # model input (or a tuple)
        output_path,                          # where to save the model
        export_params=True,                   # store the trained parameter weights inside the model file
        input_names=['input_ids', 'attention_mask'],   # the model's input names
        output_names=['logits'],             # the model's output names
        opset_version=11                     # ONNX opset version
    )


def measure_inference_time_onnx(onnx_model_path, tokenizer, text_list, max_length=128):
    """
    Measures average inference time on an ONNX model using ONNX Runtime.
    """
    # Load ONNX model
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    start_time = time.time()

    for text in text_list:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].numpy()
        attention_mask = inputs['attention_mask'].numpy()

        session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

    end_time = time.time()

    avg_inference_time = (end_time - start_time) / len(text_list)
    return avg_inference_time


def evaluate_model_onnx(onnx_model_path, tokenizer, texts, labels, max_length=128):
    """
    Evaluate the ONNX model on a given dataset and compute accuracy, F1.
    """
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    preds = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].numpy()
        attention_mask = inputs['attention_mask'].numpy()

        logits = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })[0]

        preds.append(np.argmax(logits, axis=1)[0])

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    report = classification_report(labels, preds)
    return acc, f1, report
