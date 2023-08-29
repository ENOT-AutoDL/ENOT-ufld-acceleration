import argparse
from pathlib import Path

import torch
from torch import nn

from utils.common import get_model
from utils.config import Config


class TiCompatibleClsLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.conv = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.out_features,
            kernel_size=(1, 1),
        )

        with torch.no_grad():
            self.conv.weight.copy_(linear.weight.unsqueeze(-1).unsqueeze(-1))
            self.conv.bias.copy_(linear.bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.reshape(1, self.in_features, 1, 1)
        out_tensor = self.conv(input_tensor)
        return out_tensor


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    default_config_path = current_dir / "configs/tusimple_res18.py"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_onnx",
        type=str,
        required=True,
        help="path to output onnx",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="path to model config",
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
    parser.add_argument(
        "--ti_compatible",
        action="store_true",
        default=False,
        help="replace last Linear with conv1x1 for Texas Instruments compatibility",
    )
    parser.add_argument("--opset-version", type=int, default=9, help="opset version")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")["model_ckpt"]
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        config = Config.fromfile(args.config)
        model = get_model(config).eval().cpu()
        model.load_state_dict(checkpoint, strict=False)

    if args.ti_compatible:
        model.cls[3] = TiCompatibleClsLinear(linear=model.cls[3])

    torch.onnx.export(
        f=args.output_onnx,
        model=model,
        args=torch.ones(
            [1, 3, args.height, args.width],
            dtype=torch.float32,
        ),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset_version,
    )
