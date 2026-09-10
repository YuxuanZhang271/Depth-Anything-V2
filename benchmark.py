#!/usr/bin/env python3
"""
Full benchmark for Depth Anything V2:
- Methods: torch, tensorrt
- Encoders: vits, vitb, vitl
- Dtype (TRT): fp16, fp32

Measures average inference time per frame (after 10 warm-up frames)
and peak GPU memory usage (via pynvml).
"""

import argparse
import time
import numpy as np
import cv2
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from depth_anything_v2.dpt import DepthAnythingV2
import os

# ---------- Configuration ----------
FRAME_HEIGHT = 480
FRAME_WIDTH = 640          # typical 480p
INPUT_SIZE = 518
WARMUP = 10
TEST = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = './checkpoints'
ENGINE_PATTERN = os.path.join(CHECKPOINT_DIR, 'depth_anything_v2_{encoder}_{dtype}.engine')
MODEL_PATTERN = os.path.join(CHECKPOINT_DIR, 'depth_anything_v2_{encoder}.pth')

# Model configs (same as in run_video.py)
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# ---------- TensorRT inference class (same as run_video_trt.py) ----------
class TRTInference:
    def __init__(self, engine_path, input_size):
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        input_shape = (1, 3, input_size, input_size)
        self.context.set_input_shape(self.input_name, input_shape)

        in_shape = self.context.get_tensor_shape(self.input_name)
        out_shape = self.context.get_tensor_shape(self.output_name)

        def volume(shape):
            vol = 1
            for d in shape:
                vol *= d
            return vol

        self.input_volume = volume(in_shape)
        self.output_volume = volume(out_shape)

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.host_input = cuda.pagelocked_empty(self.input_volume, dtype=self.input_dtype)
        self.host_output = cuda.pagelocked_empty(self.output_volume, dtype=self.output_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name == self.input_name:
                self.context.set_tensor_address(name, int(self.device_input))
            elif name == self.output_name:
                self.context.set_tensor_address(name, int(self.device_output))
            else:
                raise RuntimeError(f"Unexpected tensor: {name}")

        self.stream = cuda.Stream()

    def infer(self, input_image):
        # Resize and preprocess exactly as in run_video_trt.py
        resized = cv2.resize(input_image, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        chw = np.transpose(normalized, (2, 0, 1))[None, ...]

        np.copyto(self.host_input, chw.ravel())
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        depth = self.host_output.reshape(1, 1, self.input_size, self.input_size)
        depth = np.squeeze(depth, axis=0)  # (1, H, W)
        depth = np.squeeze(depth, axis=0)  # (H, W)
        return depth

# ---------- Benchmarking ----------
def get_gpu_memory(handle):
    """Return used GPU memory in MB (via PyTorch, no NVML required)."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) // (1024 * 1024)

def generate_frame(height=FRAME_HEIGHT, width=FRAME_WIDTH):
    """Generate a random BGR frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

def benchmark_torch(encoder, handle):
    """Benchmark PyTorch model."""
    print(f"Loading torch model {encoder}...")
    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(MODEL_PATTERN.format(encoder=encoder), map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Baseline memory after model load
    baseline = get_gpu_memory(handle)
    print(f"  Baseline memory: {baseline} MB")

    # Warm-up
    for _ in range(WARMUP):
        frame = generate_frame()
        with torch.no_grad():
            _ = model.infer_image(frame, INPUT_SIZE)

    # Test
    times = []
    mem_peaks = []
    for _ in range(TEST):
        frame = generate_frame()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.infer_image(frame, INPUT_SIZE)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
        # Record memory after inference
        mem_peaks.append(get_gpu_memory(handle))

    avg_time = np.mean(times)
    peak_mem = max(mem_peaks) - baseline
    return avg_time, peak_mem

def benchmark_trt(encoder, dtype, handle):
    """Benchmark TensorRT model."""
    engine_path = ENGINE_PATTERN.format(encoder=encoder, dtype=dtype)
    print(f"Loading TRT engine {engine_path}...")
    model = TRTInference(engine_path, INPUT_SIZE)

    # Baseline memory after engine load (buffers allocated)
    baseline = get_gpu_memory(handle)
    print(f"  Baseline memory: {baseline} MB")

    # Warm-up
    for _ in range(WARMUP):
        frame = generate_frame()
        _ = model.infer(frame)

    # Test
    times = []
    mem_peaks = []
    for _ in range(TEST):
        frame = generate_frame()
        start = time.perf_counter()
        _ = model.infer(frame)
        cuda.Context.synchronize()  # ensure all operations done
        end = time.perf_counter()
        times.append((end - start) * 1000)
        mem_peaks.append(get_gpu_memory(handle))

    avg_time = np.mean(times)
    peak_mem = max(mem_peaks) - baseline
    return avg_time, peak_mem

def main():
    # GPU memory is read via torch.cuda.mem_get_info(); NVML handle not needed
    handle = None

    # Results storage
    results = []

    encoders = ['vits', 'vitb', 'vitl']

    # 1. PyTorch
    for enc in encoders:
        print(f"\n=== Benchmarking torch + {enc} ===")
        avg_time, peak_mem = benchmark_torch(enc, handle)
        results.append({
            'encoder': enc,
            'method': 'torch',
            'dtype': '-',
            'avg_time_ms': avg_time,
            'peak_mem_mb': peak_mem
        })

    # 2. TensorRT fp16 & fp32
    for enc in encoders:
        for dtype in ['fp16', 'fp32']:
            print(f"\n=== Benchmarking tensorrt + {enc} + {dtype} ===")
            try:
                avg_time, peak_mem = benchmark_trt(enc, dtype, handle)
                results.append({
                    'encoder': enc,
                    'method': 'tensorrt',
                    'dtype': dtype,
                    'avg_time_ms': avg_time,
                    'peak_mem_mb': peak_mem
                })
            except FileNotFoundError:
                print(f"  Engine not found for {enc} {dtype}, skipping.")
            except Exception as e:
                print(f"  Error: {e}")

    # Print summary table
    print("\n\n========== Summary ==========")
    print(f"{'Encoder':<8} {'Method':<10} {'Dtype':<6} {'Avg Time (ms)':<15} {'Peak Mem (MB)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['encoder']:<8} {r['method']:<10} {r['dtype']:<6} {r['avg_time_ms']:<15.2f} {r['peak_mem_mb']:<15.2f}")
    print("==============================\n")

if __name__ == '__main__':
    main()