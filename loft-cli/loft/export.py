import os
import subprocess
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_export(model_dir, format, output_dir, opset: int = 17):
    os.makedirs(output_dir, exist_ok=True)

    if format.lower() == "onnx":
        print(f"🔄 Exporting model from {model_dir} to ONNX format in {output_dir}...")

        # ✅ output_dir is a positional argument
        cmd = [
            "optimum-cli", "export", "onnx",
            "--model", model_dir,
            "--opset", str(opset),
            "--task", "text-generation",
            "--device", "cpu",
            output_dir  # ✅ positional, not --output or --output_dir
        ]

        try:
            subprocess.run(cmd, check=True)
            print("✅ ONNX export complete!")

        except subprocess.CalledProcessError as e:
            print(f"[❌] ONNX export failed. Error:\n{e}")

    else:
        print(f"[❌] Format '{format}' not supported yet.")
