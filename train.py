import argparse


from utils import *
from trainer import Trainer
from transformers import AutoConfig



parser = argparse.ArgumentParser(
    description="To train the CrossCoder choose default (Gemma base vs finetuned), scale (Pythia 160m vs 70m), or distilliation (BERT 110 vs 66m)."
)
parser.add_argument(
    "--method",
    choices=["default", "finetuning", "scale", "distillation", "same"],
    default="default",
    help="Which training method to run: 'default' for base/instruction tuned, 'scale' for different scale (trained independently), 'distillation' for student/teacher."
)
parser.add_argument(
    "--layer",
    choices=["first", "middle", "last"],
    help="Which layers to compare [scale and distillation experiment only]: 'early' for first layers, 'middle' for middle layers, or 'late' for last layers."
)
args = parser.parse_args()



device = 'cuda:0'
# method = 'distillation' # default, distillation # <<<< Now a command line argument
method = args.method
layer = args.layer

if method == 'distillation':
    ### ============= Load two (or more) models to compare ==============
    
    # 1) Base model: GPT-2 (137 M)
    teacher = HookedTransformer.from_pretrained(
        "gpt2",
        revision="main",
        device=device,
    )

    # 2) distilled student (vanilla, so no jacobian matching): DistilGPT2 (88 M)
    student = HookedTransformer.from_pretrained(
        "distilgpt2",
        revision="main",
        device=device,
    )

    print(teacher)   
    print(student)

    if layer == "first":
        ### [EXPERIMENT] Early layer comparison
        hookpoint_large = "blocks.0.hook_resid_pre"  # large model - 12 transformer blocks
        hookpoint_small =  "blocks.0.hook_resid_pre"  # small model - 6 transformer blocks
    elif layer == "middle":
        ### [EXPERIMENT] Middle
        hookpoint_large = "blocks.5.hook_resid_pre"
        hookpoint_small = "blocks.2.hook_resid_pre"
    elif layer == "last":
        ### [EXPERIMENT] Last layer comparison
        hookpoint_large = "blocks.11.hook_resid_pre"
        hookpoint_small = "blocks.5.hook_resid_pre"

    distillation_cfg = {
        "seed": 49,
        "batch_size": 1024, #updated from 4096
        "buffer_mult": 128,#reduced from 128
        "model_batch_size": 4, #updated
        "lr": 5e-5,
        "num_tokens": 200_000_000, #trained until 200 million (resampled) tokens have been seen
        "l1_coeff": 2, # sparsity penalty weight, encourages that crosscoder uses low nr of features (loss balances reconstruction quality loss with this feature sparsity loss)
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in_A": teacher.cfg.d_model, #hidden size /dim of the residual‐stream vector teacher (model A)
        "d_in_B": student.cfg.d_model, #hidden size /dim of the residual‐stream vector student (model B)
        "dict_size": 2**14, # crosscoder latent features / nr of latent features crosscoder distinguishes between = crosscoder width. crosscoder decoder dims are (d_hidden=latent features 16384, n_models=2, d_model=residual stream dim 2304) << so two decoders, one for each model
        "seq_len": 1024,
        "enc_dtype": "fp32", # "bf16", 
        "model_name": "gpt2-137m-88m", #only for logging in checkpoints
        "site": "resid_pre",
        "device": "cuda:0",
        "log_every": 100, #updated from 100
        "save_every": 10000, #updated from 30000
        "dec_init_norm": 0.08,
        # "hook_point": "blocks.5.hook_resid_pre",
        "hook_point_A": hookpoint_large,  # large model - 12 transformer blocks
        "hook_point_B": hookpoint_small,  # small model - 6 transformer blocks
        "wandb_project": "crosscoders",
        "wandb_entity": "lisann-becker-university-of-amsterdam",
        "method": "distillation",
    }

    cfg = arg_parse_update_cfg(distillation_cfg)
    all_tokens = load_tokens(cfg['model_name'])

    trainer = Trainer(cfg, teacher, student, all_tokens, method)

elif method == 'scale':
    ### ============= Load two (or more) models to compare ==============
    large_model = HookedTransformer.from_pretrained( 
        "EleutherAI/pythia-160m", 
        device=device, 
    )

    small_model = HookedTransformer.from_pretrained( 
        "EleutherAI/pythia-70m",
        device=device, 
    )

    print(small_model)
    print(large_model)

    if layer == "first":
        ### [EXPERIMENT] Early layer comparison
        hookpoint_large = "blocks.0.hook_resid_pre"  # large model - 12 transformer blocks
        hookpoint_small =  "blocks.0.hook_resid_pre"  # small model - 6 transformer blocks
    elif layer == "middle":
        ### [EXPERIMENT] Middle
        hookpoint_large = "blocks.5.hook_resid_pre"
        hookpoint_small = "blocks.2.hook_resid_pre"
    elif layer == "last":
        ### [EXPERIMENT] Last layer comparison
        hookpoint_large = "blocks.11.hook_resid_pre"
        hookpoint_small = "blocks.5.hook_resid_pre"


    scale_cfg = {
        "seed": 49,
        "batch_size": 1024, #updated from 4096
        "buffer_mult": 128,#reduced from 128
        "model_batch_size": 4, #updated
        "lr": 5e-5,
        "num_tokens": 200_000_000, #tained until 400 million (resampled) tokens have been seen
        "l1_coeff": 2, # sparsity penalty weight, encourages that crosscoder uses low nr of features (loss balances reconstruction quality loss with this feature sparsity loss)
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in_A": large_model.cfg.d_model, #hidden size /dim of the residual‐stream vector teacher (model A)
        "d_in_B": small_model.cfg.d_model, #hidden size /dim of the residual‐stream vector student (model B)
        "dict_size": 2**14, # crosscoder latent features / nr of latent features crosscoder distinguishes between = crosscoder width. crosscoder decoder dims are (d_hidden=latent features 16384, n_models=2, d_model=residual stream dim 2304) << so two decoders, one for each model
        "seq_len": 1024,
        "enc_dtype": "fp32", # "bf16", 
        "model_name": "pythia-160m-70m", #only for logging in checkpoints
        "site": "resid_pre",
        "device": "cuda:0",
        "log_every": 200, #updated from 100
        "save_every": 10000, #updated fro 30000
        "dec_init_norm": 0.08,
        # "hook_point": "blocks.5.hook_resid_pre", # residual stream of 6 layer transforer - post–add&norm vector
        ### [EXPERIMENT] comparison at different layers
        "hook_point_A": hookpoint_large,  # large model - 12 transformer blocks
        "hook_point_B": hookpoint_small,  # small model - 6 transformer blocks
        "wandb_project": "crosscoders",
        "wandb_entity": "lisann-becker-university-of-amsterdam",
        "method": "scale",

    }

    cfg = arg_parse_update_cfg(scale_cfg)
    all_tokens = load_tokens(cfg['model_name'])

    trainer = Trainer(cfg, large_model, small_model, all_tokens, method)

elif method in ['default', 'finetuning']:
    ### ============= Load two (or more) models to compare ==============
    base_model = HookedTransformer.from_pretrained(
        "gemma-2-2b", 
        device=device, 
    )

    chat_model = HookedTransformer.from_pretrained(
        "gemma-2-2b-it", 
        device=device, 
    )

    print(base_model)
    print(chat_model)

    if layer == "first":
        ### [EXPERIMENT] Early layer comparison
        hookpoint = "blocks.0.hook_resid_pre"  # large model - 26 transformer blocks
    elif layer == "middle":
        ### [EXPERIMENT] Middle
        hookpoint = "blocks.13.hook_resid_pre"
    elif layer == "last":
        ### [EXPERIMENT] Last layer comparison
        hookpoint = "blocks.25.hook_resid_pre"


    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "buffer_mult": 8,#reduced from 128
        "model_batch_size": 1, #updated
        "lr": 5e-5,
        "num_tokens": 400_000_000,
        "l1_coeff": 2, # sparsity penalty weight, encourages that crosscoder uses low nr of features (loss balances reconstruction quality loss with this feature sparsity loss)
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in": base_model.cfg.d_model, #hidden size /dim of the residual‐stream vector TODO adapt - different for student and teacher, needs projection
        "dict_size": 2**14, # crosscoder latent features / nr of latent features crosscoder distinguishes between = crosscoder width. crosscoder decoder dims are (d_hidden=latent features 16384, n_models=2, d_model=residual stream dim 2304) << so two decoders, one for each model
        "seq_len": 1024,
        "enc_dtype": "fp32", # "bf16", 
        "model_name": "gemma-2-2b",
        "site": "resid_pre",
        "device": "cuda:0",
        "log_every": 500, #updated from 100
        "save_every": 10000, #updated from 30000
        "dec_init_norm": 0.08,
        "hook_point": hookpoint, # residual stream of 14 layer transformer - post–add&norm vector
        "wandb_project": "crosscoders",
        "wandb_entity": "lisann-becker-university-of-amsterdam",
        "method": "finetuning",
    }


    cfg = arg_parse_update_cfg(default_cfg)
    all_tokens = load_tokens(cfg['model_name'])


    trainer = Trainer(cfg, base_model, chat_model, all_tokens)

elif method == 'same':
    ### ============= Load two (or more) models to compare ==============
    model = HookedTransformer.from_pretrained( 
        "gpt2", # EleutherAI/pythia-70m EleutherAI/pythia-160m gpt2 distilgpt2
        revision="main",
        device=device, 
    )

    print(model)

    # if layer == "first":
        ### [EXPERIMENT] Early layer comparison
        # hookpoint = "blocks.0.hook_resid_pre"  # large model - 12 transformer blocks
        # hookpoint =  "blocks.0.hook_resid_pre"  # small model - 6 transformer blocks
    if layer == "middle":
        ### [EXPERIMENT] Middle
        hookpoint = "blocks.5.hook_resid_pre"
        # hookpoint = "blocks.2.hook_resid_pre"
    # elif layer == "last":
        ### [EXPERIMENT] Last layer comparison
        # hookpoint = "blocks.11.hook_resid_pre"
        # hookpoint = "blocks.5.hook_resid_pre"


    default_cfg = {
        "seed": 49,
        "batch_size": 1024, #updated from 4096
        "buffer_mult": 128,#reduced from 128
        "model_batch_size": 4, #updated
        "lr": 5e-5,
        "num_tokens": 200_000_000,
        "l1_coeff": 2, # sparsity penalty weight, encourages that crosscoder uses low nr of features (loss balances reconstruction quality loss with this feature sparsity loss)
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in": model.cfg.d_model, #hidden size /dim of the residual‐stream vector TODO adapt - different for student and teacher, needs projection
        "dict_size": 2**14, # crosscoder latent features / nr of latent features crosscoder distinguishes between = crosscoder width. crosscoder decoder dims are (d_hidden=latent features 16384, n_models=2, d_model=residual stream dim 2304) << so two decoders, one for each model
        "seq_len": 1024,
        "enc_dtype": "fp32", # "bf16", 
        "model_name": "gpt2-137m", #pythia-160m pythia-70m gpt2-137m gpt2-88m
        "site": "resid_pre",
        "device": "cuda:0",
        "log_every": 500, #updated from 100
        "save_every": 10000, #updated from 30000
        "dec_init_norm": 0.08,
        "hook_point": hookpoint, # residual stream of 14 layer transformer - post–add&norm vector
        "wandb_project": "crosscoders",
        "wandb_entity": "lisann-becker-university-of-amsterdam",
        "method": "same",
    }


    cfg = arg_parse_update_cfg(default_cfg)
    all_tokens = load_tokens(cfg['model_name'])


    trainer = Trainer(cfg, model, model, all_tokens)


trainer.train()
