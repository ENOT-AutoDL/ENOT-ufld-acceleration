import argparse

import fvcore
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True, help="Path to model")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(800, 320),
        help="Size of image for MACs measurement",
    )

    args = parser.parse_args()

    model = torch.load(args.model_ckpt, map_location="cpu")["model_ckpt"]

    counter = fvcore.nn.FlopCountAnalysis(model=model, inputs=torch.ones(1, 3, *args.image_size))

    print(counter.total())
