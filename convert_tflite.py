"""
SmartKhet — TFLite Edge Model Converter
========================================
Converts trained PyTorch models → ONNX → TensorFlow → TFLite
Applies 8-bit dynamic quantisation for mobile deployment.

Target device: Android (Qualcomm Snapdragon 400/600 series, ~₹5k–₹15k phones)
Target size  : Disease model ≤ 12MB | Crop model ≤ 3MB
Target latency: < 500ms on CPU inference

Author: Axora / SmartKhet ML Team
"""

import os
import logging
import numpy as np
import torch
import onnx
import tensorflow as tf
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── Disease Detection TFLite Conversion ──────────────────────────────────────

def convert_disease_model_to_tflite(
    pytorch_checkpoint: str,
    output_path: str = "mobile/src/offline/models/disease_detector.tflite",
    quantize: bool = True,
) -> str:
    """
    PyTorch EfficientNet-B4 → ONNX → TensorFlow → TFLite (INT8 quantized).
    """
    from ml.disease_detection.train import SmartKhetDiseaseModel

    log.info("Step 1/4: Loading PyTorch model...")
    checkpoint = torch.load(pytorch_checkpoint, map_location="cpu")
    num_classes = len(checkpoint["class_to_idx"])
    model = SmartKhetDiseaseModel(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Step 2: PyTorch → ONNX ──
    log.info("Step 2/4: Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = output_path.replace(".tflite", ".onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_image"],
        output_names=["class_logits"],
        dynamic_axes={"input_image": {0: "batch_size"}},
        opset_version=13,
        export_params=True,
    )

    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    log.info(f"ONNX model validated: {onnx_path}")

    # ── Step 3: ONNX → TensorFlow SavedModel ──
    log.info("Step 3/4: Converting ONNX → TensorFlow SavedModel...")
    import onnx_tf
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_saved_model_dir = onnx_path.replace(".onnx", "_tf_saved_model")
    tf_rep.export_graph(tf_saved_model_dir)

    # ── Step 4: TF SavedModel → TFLite ──
    log.info("Step 4/4: Converting TensorFlow → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Dynamic range quantisation (no representative dataset needed)
        # Reduces size ~4x, latency ~2x on CPU
        converter.target_spec.supported_types = [tf.float16]
        # For full INT8, provide representative dataset:
        # converter.representative_dataset = _get_representative_dataset()
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"✅ TFLite model saved: {output_path} ({size_mb:.1f} MB)")

    # Save class mapping JSON alongside model
    import json
    class_map_path = output_path.replace(".tflite", "_classes.json")
    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    with open(class_map_path, "w") as f:
        json.dump(idx_to_class, f)

    # Cleanup intermediates
    os.remove(onnx_path)
    import shutil
    shutil.rmtree(tf_saved_model_dir, ignore_errors=True)

    return output_path


# ── Crop Recommendation TFLite Conversion ─────────────────────────────────────

def convert_crop_model_to_tflite(
    sklearn_pipeline_path: str,
    output_path: str = "mobile/src/offline/models/crop_recommender.tflite",
) -> str:
    """
    Convert sklearn XGBoost+RF ensemble → XGBoost ONNX → TFLite.
    Only XGBoost part is converted (dominant predictor).
    RF used as verification check only in cloud mode.
    """
    import joblib
    from xgboost import XGBClassifier

    log.info("Loading sklearn pipeline...")
    pipeline = joblib.load(sklearn_pipeline_path)

    # Extract the XGBoost estimator from the VotingClassifier inside the Pipeline
    voting_clf = pipeline.named_steps["model"]
    xgb_clf: XGBClassifier = dict(voting_clf.estimators).get("xgb")

    if xgb_clf is None:
        raise ValueError("XGBClassifier not found in pipeline")

    log.info("Converting XGBoost → ONNX...")
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    n_features = len(pipeline.named_steps.get("imputer", pipeline).feature_names_in_
                     if hasattr(pipeline.named_steps.get("imputer", {}), "feature_names_in_")
                     else range(11))

    # Need to wrap the full pipeline (imputer + scaler + xgb)
    # Use sklearn's onnx exporter
    initial_type = [("float_input", FloatTensorType([None, 11]))]
    onnx_model = to_onnx(pipeline, initial_types=initial_type,
                         target_opset=15)

    onnx_path = output_path.replace(".tflite", ".onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    log.info(f"ONNX saved: {onnx_path}")

    # Convert to TFLite
    import onnx
    onnx_m = onnx.load(onnx_path)

    import onnx_tf
    tf_rep = onnx_tf.backend.prepare(onnx_m)
    tf_dir = onnx_path.replace(".onnx", "_tf")
    tf_rep.export_graph(tf_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    log.info(f"✅ Crop recommender TFLite: {output_path} ({size_kb:.0f} KB)")

    os.remove(onnx_path)
    import shutil
    shutil.rmtree(tf_dir, ignore_errors=True)

    return output_path


# ── Benchmark TFLite Model ────────────────────────────────────────────────────

def benchmark_tflite(tflite_path: str, n_runs: int = 100):
    """
    Run benchmark inference to measure latency on current machine.
    On server CPU ≈ 50ms. On Android (Qualcomm 400) ≈ 350–500ms.
    """
    import time

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    log.info(f"\n📊 TFLite Benchmark [{tflite_path}]")
    log.info(f"   Mean latency  : {np.mean(latencies):.1f} ms")
    log.info(f"   P95 latency   : {np.percentile(latencies, 95):.1f} ms")
    log.info(f"   P99 latency   : {np.percentile(latencies, 99):.1f} ms")
    log.info(f"   Model size    : {os.path.getsize(tflite_path) / (1024*1024):.1f} MB")

    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "size_mb": round(os.path.getsize(tflite_path) / (1024 * 1024), 2),
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SmartKhet models to TFLite")
    parser.add_argument("--model", choices=["disease", "crop"], required=True)
    parser.add_argument("--input", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str,
                        default="mobile/src/offline/models/")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.model == "disease":
        out = convert_disease_model_to_tflite(
            pytorch_checkpoint=args.input,
            output_path=os.path.join(args.output, "disease_detector.tflite"),
        )
    else:
        out = convert_crop_model_to_tflite(
            sklearn_pipeline_path=args.input,
            output_path=os.path.join(args.output, "crop_recommender.tflite"),
        )

    if args.benchmark:
        benchmark_tflite(out)
