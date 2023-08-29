import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")

    args = parser.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")

    if isinstance(ckpt["model"], torch.nn.Module):
        ckpt["model_ckpt"] = ckpt["model"]
    else:
        raise ValueError("model key is not nn.Module")

    torch.save(ckpt, args.model_path)
