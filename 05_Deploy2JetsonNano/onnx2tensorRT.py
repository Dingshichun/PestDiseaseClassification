import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, precision="fp16"):
    with trt.Builder(TRT_LOGGER) as builder:
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(network_flags) as network:
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # 设置动态形状配置文件
            profile = builder.create_optimization_profile()
            profile.set_shape("input", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
            config.add_optimization_profile(profile)

            engine = builder.build_serialized_network(network, config)
            with open(engine_path, "wb") as f:
                f.write(engine)
            return engine

build_engine("shufflenet_sim.onnx", "shufflenet.engine", precision="fp16")