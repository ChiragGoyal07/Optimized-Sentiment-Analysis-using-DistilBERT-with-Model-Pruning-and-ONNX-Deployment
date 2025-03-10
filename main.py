import os
import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from utils import (
    compute_metrics,
    prune_model,
    remove_pruning_reparam,
    measure_inference_time_pytorch,
    dynamic_quantize_model,
    export_to_onnx,
    measure_inference_time_onnx,
    evaluate_model_onnx
)


def main():
    # --------------------
    # Configurations
    # --------------------
    model_name = "distilbert-base-uncased"
    output_dir = "./distilbert-sentiment"
    pruning_amount = 0.2  # 20% of weights pruned
    max_train_samples = 20000  # for demo; use full dataset for better performance
    max_val_samples = 5000     # for demo
    batch_size = 16
    num_epochs = 2
    onnx_model_path = "distilbert_pruned_quantized.onnx"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------
    # Load Dataset
    # --------------------
    print("Loading the IMDb dataset...")
    dataset = load_dataset("imdb")
    
    # For speed in demonstration, optionally slice the dataset
    train_dataset = dataset["train"].shuffle(seed=42).select(range(max_train_samples))
    val_dataset = dataset["test"].shuffle(seed=42).select(range(max_val_samples))

    # --------------------
    # Tokenizer
    # --------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    # Convert to PyTorch format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # --------------------
    # Model Initialization
    # --------------------
    print("Initializing DistilBertForSequenceClassification...")
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # --------------------
    # Training Setup
    # --------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # --------------------
    # Train Model
    # --------------------
    print("Training the model...")
    trainer.train()

    # Evaluate baseline model
    print("Evaluating the model before pruning and quantization...")
    baseline_metrics = trainer.evaluate()
    print(f"Baseline metrics: {baseline_metrics}")

    # Measure baseline inference time (PyTorch)
    sample_texts = [
        "The movie was absolutely wonderful!",
        "I really did not like this film.",
        "It was an average experience, could be better."
    ]
    baseline_inference_time = measure_inference_time_pytorch(model, tokenizer, sample_texts, device)
    print(f"Baseline PyTorch average inference time per sample: {baseline_inference_time:.6f} seconds")

    # --------------------
    # Model Pruning
    # --------------------
    print(f"Pruning model with {pruning_amount*100}% unstructured magnitude-based pruning...")
    model = prune_model(model, amount=pruning_amount)
    model = remove_pruning_reparam(model)

    # Evaluate pruned model
    print("Evaluating the model after pruning...")
    pruned_metrics = trainer.evaluate()
    print(f"Pruned model metrics: {pruned_metrics}")

    pruned_inference_time = measure_inference_time_pytorch(model, tokenizer, sample_texts, device)
    print(f"Pruned PyTorch average inference time per sample: {pruned_inference_time:.6f} seconds")

    # --------------------
    # Model Quantization
    # --------------------
    print("Applying dynamic quantization to pruned model (for CPU inference only)...")
    # Move model to CPU for quantization
    model.cpu()
    quantized_model = dynamic_quantize_model(model)

    # Evaluate quantized model on CPU
    print("Evaluating the quantized model on CPU...")
    trainer.model = quantized_model  # so we can reuse the trainer's evaluate method (though it's on CPU only)
    quantized_metrics = trainer.evaluate()
    print(f"Quantized model metrics: {quantized_metrics}")

    # Measure quantized inference time
    quantized_inference_time = measure_inference_time_pytorch(quantized_model, tokenizer, sample_texts, device='cpu')
    print(f"Quantized PyTorch average inference time (CPU) per sample: {quantized_inference_time:.6f} seconds")

    # --------------------
    # Export to ONNX
    # --------------------
    print("Exporting quantized model to ONNX...")
    export_to_onnx(quantized_model, tokenizer, output_path=onnx_model_path, max_length=128, device='cpu')
    print(f"ONNX model saved to {onnx_model_path}")

    # --------------------
    # ONNX Inference
    # --------------------
    print("Measuring inference time with ONNX Runtime...")
    onnx_inference_time = measure_inference_time_onnx(onnx_model_path, tokenizer, sample_texts, max_length=128)
    print(f"ONNX average inference time per sample: {onnx_inference_time:.6f} seconds")

    # Evaluate ONNX model
    print("Evaluating ONNX model on validation dataset...")
    val_texts = [val_dataset[i]["text"] for i in range(len(val_dataset))]
    val_labels = [val_dataset[i]["label"] for i in range(len(val_dataset))]
    onnx_acc, onnx_f1, onnx_clf_report = evaluate_model_onnx(onnx_model_path, tokenizer, val_texts, val_labels, max_length=128)

    print(f"ONNX Model Accuracy: {onnx_acc:.4f}, F1: {onnx_f1:.4f}")
    print("Classification Report:\n", onnx_clf_report)

    # --------------------
    # Model Size Comparison
    # --------------------
    def model_size_mb(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)

    # Original PyTorch model (checkpoint) size
    # We can save the model to measure
    baseline_model_path = "distilbert_baseline.pt"
    torch.save(trainer.model.state_dict(), baseline_model_path)
    baseline_size = model_size_mb(baseline_model_path)

    # Quantized + pruned PyTorch model
    quantized_model_path = "distilbert_pruned_quantized.pt"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    quantized_size = model_size_mb(quantized_model_path)

    # ONNX model size
    onnx_size = model_size_mb(onnx_model_path)

    print("\nModel size comparison (MB):")
    print(f"Baseline PyTorch model: {baseline_size:.2f} MB")
    print(f"Pruned & Quantized PyTorch model: {quantized_size:.2f} MB")
    print(f"ONNX model: {onnx_size:.2f} MB")

    print("\n=== Final Results ===")
    print("Baseline Model Metrics:", baseline_metrics)
    print("Pruned Model Metrics:", pruned_metrics)
    print("Quantized Model Metrics:", quantized_metrics)
    print(f"ONNX Model Accuracy: {onnx_acc:.4f}, F1: {onnx_f1:.4f}")

    print("\nInference Time (seconds/sample):")
    print(f"Baseline PyTorch (GPU/CPU)  : {baseline_inference_time:.6f}")
    print(f"Pruned PyTorch (GPU/CPU)    : {pruned_inference_time:.6f}")
    print(f"Quantized PyTorch (CPU)     : {quantized_inference_time:.6f}")
    print(f"ONNX Runtime (CPU)          : {onnx_inference_time:.6f}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
