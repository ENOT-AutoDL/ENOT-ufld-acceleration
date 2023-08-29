import argparse
import os
import pickle
import tempfile
import zipfile
from pathlib import Path

import onnx
import requests
import torch

from export import TiCompatibleClsLinear


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_ckpt", required=True, type=str, help="Path to PyTorch model ckpt")
    parser.add_argument(
        "-c",
        "--calibration_data_zip",
        required=True,
        type=str,
        help="Path to the calibration data zip archive",
    )
    parser.add_argument(
        "-o",
        "--output_model",
        required=True,
        type=str,
        help="Path to the output zip",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host name or IP address of compilation server. Default value is '0.0.0.0'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=15003,
        help="Port of compilation server. Default value is 15003",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="input image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="input image width",
    )
    parser.add_argument("--opset-version", type=int, default=9, help="opset version")
    return parser.parse_args()


def main():
    args = parse()

    model = torch.load(args.model_ckpt, map_location="cpu")["model_ckpt"]
    model.cls[3] = TiCompatibleClsLinear(linear=model.cls[3])

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_name = os.path.join(tmpdir, "temp.onnx")
        torch.onnx.export(
            f=onnx_name,
            model=model,
            args=torch.ones(
                [1, 3, args.height, args.width],
                dtype=torch.float32,
            ),
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset_version,
        )

        onnx_model = onnx.load(onnx_name).SerializeToString()

    with open(args.calibration_data_zip, "rb") as calibration_data_zip_file:
        calibration_data = calibration_data_zip_file.read()

    print("Start compilation, please wait... Compilation takes about 15 minutes (up to 1 hour for a large model)")
    response = requests.post(
        url=f"http://{args.host}:{args.port}/compile",
        data=pickle.dumps({"model": onnx_model, "calibration_data": calibration_data}),
        headers={"Content-Type": "application/octet-stream"},
        timeout=90 * 60,  # 1.5h
    )

    if response.status_code == 200:
        artifacts_dir = args.output_model
        artifacts_zip = Path(artifacts_dir).with_suffix(".zip")

        with open(artifacts_zip, "wb") as output_model_file:
            output_model_file.write(response.content)

        # Extract data
        with zipfile.ZipFile(artifacts_zip, "r") as zf:
            zf.extractall(artifacts_dir)

        print("Compiled model saved")
    else:
        raise RuntimeError(f"Expected status code is 200, got {response.status_code}; reason: {response.reason}")


if __name__ == "__main__":
    main()
