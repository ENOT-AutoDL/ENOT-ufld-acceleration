import argparse

import onnx
import torch
from enot_latency_server.client import measure_latency_remote
from onnxsim import simplify

from prune import measure_latency_on_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", default=None, help="Path to model checkpoint for latency measurement")
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(320, 800),
        help="Size of image for MACs measurement",
    )
    parser.add_argument("--host", default="localhost", type=str, help="Host of latency measurement server")
    parser.add_argument("--port", default=15003, type=int, help="Port of latency measurement server")
    parser.add_argument("--num_runs", default=1, type=int)
    parser.add_argument("--ti_server", action="store_true", help="Whether to measure on TI server.")
    parser.add_argument("--onnx", default=None, type=str, help="Path to model ONNX to measure latency.")

    args = parser.parse_args()

    if args.onnx:
        onnx_model = onnx.load(args.onnx)
        onnx_model, _ = simplify(onnx_model)
        latency = measure_latency_remote(onnx_model.SerializeToString(), host=args.host, port=args.port)
        print(latency)
        exit()

    model = torch.load(args.model_ckpt, map_location="cpu")["model_ckpt"]
    if torch.cuda.is_available():
        model = model.cuda()
    print("model_loaded")

    latency = measure_latency_on_server(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_size=args.image_size,
        port=args.port,
        host=args.host,
        ti_server=args.ti_server,
    )
    print(latency)
