import numpy as np
import faiss
import onnxruntime as ort

print("Loading FAISS...")
index = faiss.IndexFlatL2(4)
print("FAISS OK")

print("Loading ONNX...")
session = ort.InferenceSession("models/ppo_transistor.onnx", providers=['CPUExecutionProvider'])
print("ONNX OK")
