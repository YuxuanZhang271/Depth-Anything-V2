#!/usr/bin/env python3
"""
Full benchmark for Depth Anything V2:
- Methods: torch, tensorrt
- Encoders: vits, vitb, vitl
- Dtype (TRT): fp16, fp32, int8
- Variant (TRT): arbitrary tag between encoder and dtype (e.g. '3060', 'jetson')

Discovers engines / checkpoints dynamically from ./checkpoints.
Measures average inference time per frame (after warm-up) and GPU memory.
"""

import argparse
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from depth_anything_v2.dpt import DepthAnythingV2

# ---------- Configuration ----------
FRAME_HEIGHT = 480
FRAME_WIDTH = 640
INPUT_SIZE = 518
WARMUP = 10
TEST = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = './checkpoints'

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Filename convention: depth_anything_v2_<encoder>[_<variant>]_<dtype>.engine
_ENGINE_RE = re.compile(
    r'^depth_anything_v2_(?P<encoder>vits|vitb|vitl)'
    r'(?:_(?P<variant>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?))?'
    r'_(?P<dtype>fp16|fp32|int8)\.engine$'
)
_CHECKPOINT_RE = re.compile(r'^depth_anything_v2_(?P<encoder>vits|vitb|vitl)\.pth$')


# ---------- Dynamic discovery ----------
def discover_engines(checkpoint_dir, dtypes=None, variants=None, encoders=None):
    """Return list of dicts: {path, encoder, variant, dtype}. Only existing files."""
    found = []
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, '*.engine'))):
        m = _ENGINE_RE.match(Path(path).name)
        if not m:
            print(f"  [skip] Unrecognised engine name: {Path(path).name}")
            continue
        info = m.groupdict()
        info['variant'] = info['variant'] or 'default'
        info['path'] = path
        if dtypes   and info['dtype']   not in dtypes:   continue
        if variants and info['variant'] not in variants: continue
        if encoders and info['encoder'] not in encoders: continue
        found.append(info)
    return found


def discover_checkpoints(checkpoint_dir, encoders=None):
    """Return dict {encoder: path} for available .pth files."""
    found = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth'))):
        m = _CHECKPOINT_RE.match(Path(path).name)
        if not m:
            print(f"  [skip] Unrecognised checkpoint name: {Path(path).name}")
            continue
        enc = m.group('encoder')
        if encoders and enc not in encoders:
            continue
        found[enc] = path
    return found


# ---------- TensorRT inference ----------
class TRTInference:
    def __init__(self, engine_path, input_size):
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            (self.input_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
             else self.output_names).append(name)

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.context.set_input_shape(self.input_name, (1, 3, input_size, input_size))

        in_shape = self.context.get_tensor_shape(self.input_name)
        out_shape = self.context.get_tensor_shape(self.output_name)
        vol = lambda s: int(np.prod(s))

        self.input_volume = vol(in_shape)
        self.output_volume = vol(out_shape)
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.host_input = cuda.pagelocked_empty(self.input_volume, dtype=self.input_dtype)
        self.host_output = cuda.pagelocked_empty(self.output_volume, dtype=self.output_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            addr = int(self.device_input) if name == self.input_name else int(self.device_output)
            self.context.set_tensor_address(name, addr)

        self.stream = cuda.Stream()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(self, input_image):
        resized = cv2.resize(input_image, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose((rgb - self.mean) / self.std, (2, 0, 1))[None, ...]

        np.copyto(self.host_input, chw.ravel())
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        return self.host_output.reshape(self.input_size, self.input_size)


# ---------- Benchmarking ----------
def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return (total - free) // (1024 * 1024)


def generate_frame(height=FRAME_HEIGHT, width=FRAME_WIDTH):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark_torch(encoder, checkpoint_path):
    print(f"Loading torch model {encoder} from {checkpoint_path}...")
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE).eval()

    baseline = get_gpu_memory()
    for _ in range(WARMUP):
        with torch.no_grad():
            model.infer_image(generate_frame(), INPUT_SIZE)

    times, mems = [], []
    for _ in range(TEST):
        frame = generate_frame()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.infer_image(frame, INPUT_SIZE)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(get_gpu_memory())

    return float(np.mean(times)), float(max(mems) - baseline)


def benchmark_trt(engine_path):
    print(f"Loading TRT engine {engine_path}...")
    model = TRTInference(engine_path, INPUT_SIZE)

    baseline = get_gpu_memory()
    for _ in range(WARMUP):
        model.infer(generate_frame())

    times, mems = [], []
    for _ in range(TEST):
        frame = generate_frame()
        t0 = time.perf_counter()
        model.infer(frame)
        cuda.Context.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        mems.append(get_gpu_memory())

    return float(np.mean(times)), float(max(mems) - baseline)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_DIR)
    parser.add_argument('--encoders', nargs='+', default=None,
                        choices=list(MODEL_CONFIGS), help='Filter encoders')
    parser.add_argument('--dtypes', nargs='+', default=None,
                        choices=['fp16', 'fp32', 'int8'])
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Filter engine variants, e.g. 3060 jetson')
    parser.add_argument('--skip-torch', action='store_true')
    parser.add_argument('--skip-trt', action='store_true')
    args = parser.parse_args()

    # Discover
    print(f"Scanning {args.checkpoint_dir} ...")
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.encoders)
    engines = discover_engines(args.checkpoint_dir, args.dtypes, args.variants, args.encoders)

    print(f"  Found {len(checkpoints)} torch checkpoint(s): {sorted(checkpoints)}")
    print(f"  Found {len(engines)} TRT engine(s):")
    for e in engines:
        print(f"    - {Path(e['path']).name}  (enc={e['encoder']}, variant={e['variant']}, dtype={e['dtype']})")

    results, skipped = [], []

    # 1. Torch
    if not args.skip_torch:
        for enc in (args.encoders or list(MODEL_CONFIGS)):
            if enc not in checkpoints:
                skipped.append((enc, 'torch', '-', '-', 'checkpoint missing'))
                continue
            print(f"\n=== Benchmarking torch + {enc} ===")
            try:
                t, m = benchmark_torch(enc, checkpoints[enc])
                results.append(dict(encoder=enc, method='torch', dtype='-',
                                    variant='-', avg_time_ms=t, peak_mem_mb=m))
            except Exception as ex:
                skipped.append((enc, 'torch', '-', '-', f'error: {ex}'))

    # 2. TensorRT — one row per discovered engine
    if not args.skip_trt:
        for e in engines:
            print(f"\n=== Benchmarking tensorrt + {e['encoder']} + {e['variant']} + {e['dtype']} ===")
            try:
                t, m = benchmark_trt(e['path'])
                results.append(dict(encoder=e['encoder'], method='tensorrt',
                                    dtype=e['dtype'], variant=e['variant'],
                                    avg_time_ms=t, peak_mem_mb=m))
            except Exception as ex:
                skipped.append((e['encoder'], 'tensorrt', e['dtype'], e['variant'], f'error: {ex}'))

    # Summary
    print("\n\n========== Summary ==========")
    hdr = f"{'Encoder':<8} {'Method':<10} {'Dtype':<6} {'Variant':<10} {'Avg Time (ms)':<15} {'Peak Mem (MB)':<15}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['encoder']:<8} {r['method']:<10} {r['dtype']:<6} {r['variant']:<10} "
              f"{r['avg_time_ms']:<15.2f} {r['peak_mem_mb']:<15.2f}")

    if skipped:
        print("\nSkipped:")
        for s in skipped:
            print(f"  {s[0]:<8} {s[1]:<10} dtype={s[2]:<6} variant={s[3]:<10} ({s[4]})")
    print("==============================\n")


if __name__ == '__main__':
    main()