import argparse
import os
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="path to model. This is path to 'onnx file'/'artifacts dir' if you want to run model on CPU/NPU.",
    )
    parser.add_argument(
        "-i",
        "--input_data_dir",
        type=str,
        required=True,
        help="path to input data directory",
    )
    parser.add_argument(
        "-o",
        "--output_data_dir",
        type=str,
        required=True,
        help="path to output data directory",
    )
    parser.add_argument("-d", "--debug_level", type=int, default=0, help="debug level")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if model_path.is_dir():
        providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
        tidl_provider_options = {
            "platform": "J7",
            "version": "7.2",
            "debug_level": args.debug_level,
            "max_num_subgraphs": 16,
            "ti_internal_nc_flag": 1601,
            "tidl_tools_path": os.environ["TIDL_TOOLS_PATH"],
            "artifacts_folder": str(model_path),
        }
        provider_options = [tidl_provider_options, {}]

        onnx_path = list(model_path.glob("*.onnx"))
        if len(onnx_path) == 0:
            raise ValueError("cannot find model onnx in artifacts directory")
        if len(onnx_path) > 1:
            raise ValueError("artifacts directory must contain only one onnx")

        onnx_path = onnx_path[0]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
        onnx_path = model_path

    ort_session_options = ort.SessionOptions()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=providers,
        provider_options=provider_options,
        sess_options=ort_session_options,
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise NotImplementedError("Case with multiple inputs is not implemented")

    input_name = inputs[0].name
    output_names = [output.name for output in outputs]

    input_data_dir = Path(args.input_data_dir)
    output_data_dir = Path(args.output_data_dir)
    if input_data_dir == output_data_dir:
        raise ValueError("--input_data_dir and --output_data_dir cannot be the same")

    if not output_data_dir.exists():
        output_data_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    t_0 = perf_counter()
    for i, input_data_path in enumerate(input_data_dir.glob("*.pickle")):
        with input_data_path.open("rb") as data_file:
            data = pickle.load(data_file)
            data = np.expand_dims(data, axis=0)
            t_1 = perf_counter()
            result = session.run(output_names=output_names, input_feed={input_name: data})
            t_2 = perf_counter()

        output_data_path = output_data_dir / f"result_{input_data_path.name}"
        with output_data_path.open("wb") as result_file:
            pickle.dump(result, result_file)

        stats.append(t_2 - t_1)
        if (i + 1) % 100 == 0:
            total_time = perf_counter() - t_0
            total_inference_time = sum(stats)
            print("TIME STATS:")
            print(f"  AVG TIME PER FILE = {total_time * 1000.0 / len(stats)} ms")
            print(f"  AVG TIME PER RUN = {total_inference_time * 1000.0 / len(stats)} ms")
            print(f"  TOTAL TIME = {total_time} s")
            print()
