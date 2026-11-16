import torch
import argparse
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--input-size', type=int, default=518, help='Model input size (square)')
    parser.add_argument('--output', type=str, default='depth_anything_v2.onnx', help='Output ONNX file')
    args = parser.parse_args()

    # Model configuration (same as in run_video.py)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    model.eval().cuda()

    # Dummy input with fixed size
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size).cuda()

    # Export to ONNX with dynamic batch (if you want batch>1 later, you can keep batch dynamic)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={'input': {0: 'batch_size'}, 'depth': {0: 'batch_size'}},
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✅ ONNX model exported to {args.output}")