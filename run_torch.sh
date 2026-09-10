#!/bin/bash
# Runs the PyTorch version for vits, vitb, vitl on the specified video

VIDEO_PATH="../../Datasets/002_S8a_videos_480p/S01_clipped.mp4"

echo "Running torch + vits"
python run_video.py --encoder vits --video-path "$VIDEO_PATH"

echo "Running torch + vitb"
python run_video.py --encoder vitb --video-path "$VIDEO_PATH"

echo "Running torch + vitl"
python run_video.py --encoder vitl --video-path "$VIDEO_PATH"