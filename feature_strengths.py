import argparse
from buffer import Buffer
from buffer_multiscale import BufferMultiscale
from crosscoder import CrossCoder
from crosscoder_multiscale import CrossCoderMultiscale
from utils import *
import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        "Average feature strength computation"
    )
    parser.add_argument(
        "-cd",
        "--config_dir",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
    )
    parser.add_argument(
        "-A",
        "--modelA",
        type=str,
    )
    parser.add_argument(
        "-B",
        "--modelB",
        type=str,
    )
    parser.add_argument(
        "-ms",
        "--is_multiscale",
        action="store_true",
    )
    return parser.parse_args()

def load_model(model_name, device) -> HookedTransformer:
    match model_name:
        case "gpt2":
            return HookedTransformer.from_pretrained(
                "gpt2",
                revision="main",
                device=device,
            )
        case "distilgpt2":
            return HookedTransformer.from_pretrained(
                "distilgpt2",
                revision="main",
                device=device,
            )
        case "pythia-160m":
            return HookedTransformer.from_pretrained( 
                "EleutherAI/pythia-160m", 
                device=device, 
            )
        case "pythia-70m":
            return HookedTransformer.from_pretrained( 
                "EleutherAI/pythia-70m",
                device=device, 
            )
        case "gemma-2-2b":
            return HookedTransformer.from_pretrained(
                "gemma-2-2b", 
                device=device, 
            )
        case "gemma-2-2b-it":
            return HookedTransformer.from_pretrained(
                "gemma-2-2b-it", 
                device=device, 
            )
        case _:
            raise ValueError(f"Unkown model name: {model_name}")

def cca(x: torch.Tensor, y: torch.Tensor) -> float:
  """Computes the mean squared CCA correlation (R^2_{CCA}).

  Args:
    `x`: A num_examples x num_features matrix of features.
    `y`: A num_examples x num_features matrix of features.

  Returns:
    The mean squared CCA correlations between X and Y in the range [0, 1].
  """
  qx, _ = torch.linalg.qr(x - x.mean())
  qy, _ = torch.linalg.qr(y - y.mean())

  return torch.linalg.norm(qx.T @ qy).item() ** 2 / min(x.shape[1], y.shape[1])

if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    
    # load cross coder
    if args.is_multiscale:
        crosscoder = CrossCoderMultiscale.load(args.config_dir, args.version)
    else:
        crosscoder = CrossCoder.load(args.config_dir, args.version)
    cfg = crosscoder.cfg

    # load models
    modelA = load_model(args.modelA, cfg["device"])
    modelB = load_model(args.modelB, cfg["device"])

    # load data
    all_tokens = load_tokens(cfg["model_name"])

    if args.is_multiscale:
        buff = BufferMultiscale(cfg, modelA, modelB, all_tokens)
    else:
        buff = Buffer(cfg, modelA, modelB, all_tokens)
    print(f"Normalization factors for {cfg['model_name']}: {buff.normalisation_factor}")

    # iterate over tokens
    print("Computing feature strengths...")

    feature_means = torch.zeros(cfg["dict_size"], device=cfg["device"])
    total_steps = cfg["buffer_mult"]
    for i in tqdm.trange(total_steps):
        if args.is_multiscale:
            xA, xB = buff.next()
            feature_means += crosscoder.encode(xA, xB).mean(dim=0)
        else:
            x = buff.next()
            feature_means += crosscoder.encode(x).mean(dim=0)
    feature_means /= total_steps
    
    print(f"First 5 feature means: {feature_means[:5]}")
    
    # compute feature activation norms and feature similarity metric
    if args.is_multiscale:
        feature_norms_A = crosscoder.W_dec_A.norm(dim=-1)
        feature_norms_B = crosscoder.W_dec_B.norm(dim=-1)
        feature_norms_rel = feature_norms_B / (feature_norms_A + feature_norms_B)

        sim_metric = cca(crosscoder.W_dec_A, crosscoder.W_dec_B)

        path = f"features_{cfg['model_name']}_{cfg['hook_point_A']}_{cfg['hook_point_B']}.pt"
    else:
        feature_norms = crosscoder.W_dec.norm(dim=-1)
        feature_norms_rel = (feature_norms[:, 1] / feature_norms.sum(dim=-1))

        sim_metric = cca(crosscoder.W_dec[:, 0], crosscoder.W_dec[:, 1])

        path = f"features_{cfg['model_name']}_{cfg['hook_point']}.pt"

    print(f"Feature similarity metric: {sim_metric:.3f}")
    print(f"Saving to {path}...")
    torch.save({
            # how strong each feature is present in the data
            "data_strength": feature_means,
            # how strong each feature is activated in the model
            "model_strength": feature_norms_rel,
            "sim_metric": sim_metric,
        }, path
    )
