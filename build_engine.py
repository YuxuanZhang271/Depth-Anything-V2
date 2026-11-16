import os
import tensorrt as trt

ONNX_PATH   = "checkpoints/depth_anything_v2_vits.onnx"
ENGINE_PATH = "checkpoints/depth_anything_v2_vits_3060_fp16.engine"
INPUT_SIZE  = 518

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build(onnx_path: str, engine_path: str):
    builder = trt.Builder(TRT_LOGGER)

    # --- FIX 1: No EXPLICIT_BATCH flag in TensorRT 11 ---
    network = builder.create_network()

    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing {onnx_path} ...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    # Show what TensorRT sees
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  Input  '{t.name}'  shape={t.shape}  dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  Output '{t.name}'  shape={t.shape}  dtype={t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # --- FIX 2: Do NOT set BuilderFlag.FP16 in TensorRT 11 ---
    # Precision is determined by the ONNX model's tensor types.
    # To get FP16, cast the ONNX model to FP16 before parsing (see Option A above).

    # Dynamic batch: give an optimization profile
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(
        inp.name,
        min=(1, 3, INPUT_SIZE, INPUT_SIZE),
        opt=(1, 3, INPUT_SIZE, INPUT_SIZE),
        max=(4, 3, INPUT_SIZE, INPUT_SIZE),
    )
    config.add_optimization_profile(profile)

    print("Building engine (this can take a few minutes) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed (see log above)")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"✅ Saved engine -> {engine_path}")


if __name__ == "__main__":
    build(ONNX_PATH, ENGINE_PATH)