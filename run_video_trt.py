import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes CUDA context
import torch
from tqdm import tqdm

# ---------- TensorRT inference class (TensorRT 10 compatible) ----------
class TRTInference:
    def __init__(self, engine_path, input_size):
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Get input and output tensor names
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Assume single input and single output
        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]

        # Set input shape (batch=1, channels=3, height=input_size, width=input_size)
        input_shape = (1, 3, input_size, input_size)
        self.context.set_input_shape(self.input_name, input_shape)

        # Get actual shapes (may differ if dynamic)
        in_shape = self.context.get_tensor_shape(self.input_name)
        out_shape = self.context.get_tensor_shape(self.output_name)

        def volume(shape):
            vol = 1
            for d in shape:
                vol *= d
            return vol

        self.input_volume = volume(in_shape)
        self.output_volume = volume(out_shape)

        # Data types
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # Allocate pinned host and device buffers
        self.host_input = cuda.pagelocked_empty(self.input_volume, dtype=self.input_dtype)
        self.host_output = cuda.pagelocked_empty(self.output_volume, dtype=self.output_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        # Set tensor addresses in the execution context (required for v3 API)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name == self.input_name:
                self.context.set_tensor_address(name, int(self.device_input))
            elif name == self.output_name:
                self.context.set_tensor_address(name, int(self.device_output))
            else:
                raise RuntimeError(f"Unexpected tensor: {name}")

        # Create CUDA stream for async operations
        self.stream = cuda.Stream()

    def infer(self, input_image):
        """
        input_image: numpy array (H, W, 3) in BGR order (as from cv2)
        Returns: depth map (H, W) as float32 (values are raw, not normalized)
        """
        # Resize to model input size
        resized = cv2.resize(input_image, (self.input_size, self.input_size))
        # Convert to RGB and normalize to [0,1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        # Convert to CHW and add batch dimension
        chw = np.transpose(normalized, (2, 0, 1))[None, ...]  # (1, 3, H, W)

        # Copy to host input buffer
        np.copyto(self.host_input, chw.ravel())

        # Async transfer to device
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)

        # Run inference using TensorRT 10's execute_async_v3 (no bindings argument)
        self.context.execute_async_v3(self.stream.handle)

        # Async transfer back from device
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        # Reshape output to (1, 1, H, W) and squeeze to (H, W)
        depth = self.host_output.reshape(1, 1, self.input_size, self.input_size)
        depth = np.squeeze(depth, axis=0)  # (1, H, W)
        depth = np.squeeze(depth, axis=0)  # (H, W)
        return depth

# ---------- Main script ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with TensorRT')
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth_trt')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--engine', type=str, default='depth_anything_v2.engine', help='Path to TensorRT engine')
    parser.add_argument('--pred-only', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading TensorRT engine...")
    trt_model = TRTInference(args.engine, args.input_size)
    print("Engine loaded.")

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)

    margin_width = 50
    # Fix for Matplotlib < 3.6: use colormaps['Spectral_r'] directly
    cmap = matplotlib.colormaps['Spectral_r']

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_video = cv2.VideoCapture(filename)
        if not raw_video.isOpened():
            print(f"  Warning: Could not open video {filename}. Skipping.")
            continue

        frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

        if args.pred_only:
            output_width = frame_width
        else:
            output_width = frame_width * 2 + margin_width

        output_filename = os.path.splitext(os.path.basename(filename))[0] + '_' + args.encoder + '_trt.mp4'
        output_path = os.path.join(args.outdir, output_filename)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc=f"  Processing {os.path.basename(filename)}",
                    unit='frame', leave=(k == len(filenames)-1))

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            depth_small = trt_model.infer(raw_frame)  # (input_size, input_size)

            # Resize depth to original frame size
            depth = cv2.resize(depth_small, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

            # Normalize to 0-255 for visualization
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                out.write(combined_frame)

            pbar.update(1)

        pbar.close()
        raw_video.release()
        out.release()
        print(f"  Saved to {output_path}")