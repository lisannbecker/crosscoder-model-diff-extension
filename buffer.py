from utils import *
from transformer_lens import ActivationCache
import tqdm

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        # assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"]) # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
        [
            estimated_norm_scaling_factor_A,
            estimated_norm_scaling_factor_B,
        ],
        device="cuda:0",
        dtype=torch.float32,
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh_original(self):
        self.pointer = 0
        # print("Refreshing the buffer!")

        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg["model_batch_size"], num_batches
                    )
                ]
                _, cache_A = self.model_A.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_A: ActivationCache

                _, cache_B = self.model_B.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_B: ActivationCache

                acts = torch.stack([cache_A[self.cfg["hook_point"]], cache_B[self.cfg["hook_point"]]], dim=0)
                acts = acts[:, :, 1:, :] # Drop BOS
                assert acts.shape == (2, tokens.shape[0], tokens.shape[1]-1, self.model_A.cfg.d_model) # [2, batch, seq_len, d_model]
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        # print("Refreshing the buffer!")

        bs      = self.cfg["model_batch_size"]
        seq_len = self.cfg["seq_len"]
        device  = self.cfg["device"]

        # instead of bs*buffer_mult, we need exactly buffer_batches windows
        total_ctx = self.buffer_batches  
        n_tokens  = self.all_tokens.size(0)

        # sample start positions for each window of length seq_len+1
        max_start = n_tokens - (seq_len + 1)
        starts = torch.randint(0, max_start + 1, (total_ctx,), device=device)

        # build contexts: [total_ctx, seq_len+1]
        contexts = torch.stack(
            [self.all_tokens[s : s + (seq_len + 1)] for s in starts], 
            dim=0
        ).to(device)

        # inputs are the first seq_len tokens
        tokens_in = contexts[:, :-1]  # [total_ctx, seq_len]

        acts_chunks = []
        with torch.autocast(device, torch.bfloat16):
            # process in batches of size bs
            for i in range(0, total_ctx, bs):
                batch = tokens_in[i : i + bs]  # [bs, seq_len]
                _, cacheA = self.model_A.run_with_cache(
                    batch, names_filter=self.cfg["hook_point"], return_type=None
                )
                _, cacheB = self.model_B.run_with_cache(
                    batch, names_filter=self.cfg["hook_point"], return_type=None
                )
                aA = cacheA[self.cfg["hook_point"]]  # [bs, seq_len, d_model]
                aB = cacheB[self.cfg["hook_point"]]  # [bs, seq_len, d_model]

                # drop the BOS token
                acts = torch.stack([aA, aB], dim=0)[:, :, 1:, :]   # [2, bs, seq_len-1, d_model]
                # flatten bs and seq dimensions
                acts = einops.rearrange(acts, "n b s d -> (b s) n d")  # [(bs*(seq_len-1)), 2, d_model]
                acts_chunks.append(acts)

        full_acts = torch.cat(acts_chunks, dim=0).to(torch.bfloat16)
        # now full_acts.shape == (buffer_batches*(seq_len-1), 2, d_model)
        assert full_acts.shape == (self.buffer_size, 2, self.model_A.cfg.d_model), (
            f"got {full_acts.shape}, expected ({self.buffer_size},2,{self.model_A.cfg.d_model})"
        )

        # shuffle and reset pointer
        self.buffer  = full_acts[torch.randperm(self.buffer_size, device=device)]
        self.pointer = 0


    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
