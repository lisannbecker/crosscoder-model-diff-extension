
from utils import *

from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download
from pathlib import Path

from typing import NamedTuple

"""
L 
- Create checkpoints directory if does not exist to save weights at end of training
- Allow for different sized models - i.e., smaller student than teacher. TODO! need to adapt configs from d_in to d_in_A and d_in_B.
"""

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# SAVE_DIR = Path("/workspace/crosscoder-model-diff-replication/checkpoints")
# SAVE_DIR = (Path(__file__).resolve().parent / "checkpoints_distillation")





class LossOutput(NamedTuple):
    # loss: torch.Tensor
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor

class CrossCoderMultiscale(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #nr of latent features
        d_hidden = self.cfg["dict_size"]
        # d_in = self.cfg["d_in"]
        # each model's residual stream size
        d_in_A, d_in_B = self.cfg["d_in_A"], self.cfg["d_in_B"]
        init_norm  = self.cfg["dec_init_norm"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])

        ### ======================= DECODER WEIGHTS =======================
        #one decoder matrix per model: [D_hidden x D_model]
        self.W_dec_A = nn.Parameter(torch.randn(d_hidden, d_in_A, dtype=self.dtype))
        self.W_dec_B = nn.Parameter(torch.randn(d_hidden, d_in_B, dtype=self.dtype))

        # normalize each latent's decoder vector (i.e. each row) to length init_norm
        with torch.no_grad():
            self.W_dec_A.data = (
                self.W_dec_A.data
                / self.W_dec_A.data.norm(dim=1, keepdim=True)
                * init_norm
            )
            self.W_dec_B.data = (
                self.W_dec_B.data
                / self.W_dec_B.data.norm(dim=1, keepdim=True)
                * init_norm
            )

        ### ======================= ENCODER WEIGHTS =======================
        """
        Why model size does not matter
        - We have two encoders anyway (W_enc_A and W_enc_B) that map from the model's dimension, which is different in the distilling case, to the same D_hidden = nr of latent features
        - Vise versa for the two decoders
        """
        # tie encoder to decoder
        self.W_enc_A = nn.Parameter(self.W_dec_A.data.t().clone())
        self.W_enc_B = nn.Parameter(self.W_dec_B.data.t().clone())
        # hardcoding n_models to 2
        # self.W_enc = nn.Parameter(
        #     torch.empty(2, d_in, d_hidden, dtype=self.dtype)
        # )


        # self.W_dec = nn.Parameter(
        #     torch.nn.init.normal_(
        #         torch.empty(
        #             # d_hidden, 2, d_in, dtype=self.dtype
        #             d_hidden, 2, d_in_A, dtype=self.dtype
        #         )
        #     )
        # )
        # self.W_dec = nn.Parameter(
        #     torch.nn.init.normal_(
        #         torch.empty(
        #             # d_hidden, 2, d_in, dtype=self.dtype
        #             d_hidden, 2, d_in_B, dtype=self.dtype
        #         )
        #     )
        # )
        # Make norm of W_dec 0.1 for each column, separate per layer - initialise L2 norm magnitude for each feature and across the model residual stream size
        # self.W_dec.data = (
        #     self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * init_n
        # )
        # # Initialise W_enc to be the transpose of W_dec
        # self.W_enc.data = einops.rearrange(
        #     self.W_dec.data.clone(),
        #     "d_hidden n_models d_model -> n_models d_model d_hidden",
        # )


        # ======================= BIASES =======================
        self.b_enc   = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec_A = nn.Parameter(torch.zeros(d_in_A, dtype=self.dtype))
        self.b_dec_B = nn.Parameter(torch.zeros(d_in_B, dtype=self.dtype))



        # self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        # self.b_dec = nn.Parameter(
        #     torch.zeros((2, d_in), dtype=self.dtype)
        # )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, xA, xB, apply_relu=True):
        """aggregates both model streams into a single latent vector"""

        """
        x: [batch, 2, D_model]  (x[:,0] has D_A dims, x[:,1] has D_B dims)
        returns: activations [batch, D_hidden]
        """
        # x: [batch, n_models, d_model]
        # x_enc = einops.einsum( #for each batch and latent, sum over model & dimension indices to produce a (batch, d_hidden)] encoding.
        #     x,
        #     self.W_enc,
        #     "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        # )

        eA = xA @ self.W_enc_A       # [batch, D_h]
        eB = xB @ self.W_enc_B       # [batch, D_h]

        if apply_relu:
            acts = F.relu(eA + eB + self.b_enc)
        else:
            acts = (eA + eB + self.b_enc)
        return acts

    def decode(self, acts):
        """reconstructs two versions of the residual stream"""

        """
        acts: [batch, D_hidden]
        returns: reconstructions [batch, 2, D_model]
        """

        rA = acts @ self.W_dec_A + self.b_dec_A  # [batch, d_in_A]
        rB = acts @ self.W_dec_B + self.b_dec_B  # [batch, d_in_B]
        return rA, rB
        # acts_dec = einops.einsum(
        #     acts,
        #     self.W_dec,
        #     "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        # )
        # return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        xA, xB = x
        acts = self.encode(xA, xB)
        return self.decode(acts)

    def get_losses(self, x):
        """
        l2 is the reconstruction loss MSE
        l1 is the sparsity penalty

        Explained variance 1-(MSE/variance)is computed for model A and B separately

        L0 is how may features are used on average
        """
        # x: [batch, n_models, d_model]
        # x = x.to(self.dtype)
        xA, xB = x
        acts = self.encode(xA, xB) # acts: [batch, d_hidden]
        rA, rB  = self.decode(acts)

        ### ================== L2 LOSS (MSE RECONSTRUCTION) ==================
        # diff = recon.float() - x.float()
        # squared_diff = diff.pow(2)
        # l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        # l2_loss = l2_per_batch.mean()
        errA = (rA.float() - xA.float()).pow(2).sum(dim=-1)
        errB = (rB.float() - xB.float()).pow(2).sum(dim=-1)
        l2_loss = (errA + errB).mean() 


        ### ================== EXPLAINED VARIANCE PER MODEL ==================
        # total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        # explained_variance = 1 - l2_per_batch / total_variance
        varA = (xA - xA.mean(0)).pow(2).sum(dim=-1)
        varB = (xB - xB.mean(0)).pow(2).sum(dim=-1)
        eA   = 1 - errA/varA
        eB   = 1 - errB/varB
        explained_variance = 0.5*(eA + eB)


        ### ================== L1 LOSS (SPARSITY PENALTY) - weighted by decoder norms sum ==================
        # per_token_l2_loss_A = (recon[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        # total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze() # squared deviations of the raw inputs from their mean = how much activations vary
        # explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A

        # per_token_l2_loss_B = (recon[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        # total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        # explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B

        # decoder_norms = self.W_dec.norm(dim=-1) #L2 norm is magnitude of single feature across the model residual stream size, initialised to 0.8 at beginnign of training
        # once trained, each latent feature will have two decoder vectors and we compare their norms / magnitudes
        # comparison of norms: plot ratios ||Wi,A|| / ||Wi,A|| + ||Wi,B||, where i is the index of the latent feature and A and B are the two models
        # 0.5 indicates shared latents, and 0 or 1 that the latents are specific to one model

        # decoder_norms: [d_hidden, n_models]
        # total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
        # l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)
        normA = self.W_dec_A.norm(dim=1)   # [D_h]
        normB = self.W_dec_B.norm(dim=1)
        total_decoder_norm = normA + normB # [D_h]
        l1_loss = (acts * total_decoder_norm[None]).sum(-1).mean()


        ### ================== L0 PENALTY (HOW MANY LATENTS FIRE) ==================
        l0_loss = (acts>0).float().sum(-1).mean()

        # return LossOutput(l2_loss=l2_loss, l1_loss=l1_loss, l0_loss=l0_loss, explained_variance=explained_variance, explained_variance_A=explained_variance_A, explained_variance_B=explained_variance_B)
        # renamed this, minor logic changes
        return LossOutput(
            l2_loss           = l2_loss,
            l1_loss           = l1_loss,
            l0_loss           = l0_loss,
            explained_variance= explained_variance.mean(),
            explained_variance_A= eA.mean(),
            explained_variance_B= eB.mean(),
        )

    def create_save_dir(self):
        # base_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints")
        base_dir = (Path(__file__).resolve().parent / f"checkpoints_{self.cfg['method']}_{self.cfg['model_name']}/{self.cfg['hook_point_A'].replace('.','_')}__{self.cfg['hook_point_B'].replace('.','_')}")
        base_dir.mkdir(parents=True, exist_ok=True)
        version_list = [
            # int(file.name.split("_")[1])
            # for file in list(SAVE_DIR.iterdir())
            # if "version" in str(file)
            int(p.name.split("_",1)[1])
            for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("version_")
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
        path: str = "blocks.14.hook_resid_pre",
        device: Optional[Union[str, torch.device]] = None
    ) -> "CrossCoder":
        """
        Load CrossCoder weights and config from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)
            
        Returns:
            Initialized CrossCoder instance
        """

        # Download config and weights
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cfg.json"
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cc_weights.pt"
        )

        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict)

        return instance

    @classmethod
    def load(cls, version_dir, checkpoint_version):
        save_dir = Path(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(weight_path))
        return self