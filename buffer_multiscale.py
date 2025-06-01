from utils import *
from transformer_lens import ActivationCache
import tqdm

class BufferMultiscale:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        self.pointer = 0
        # assert model_A.cfg.d_model == model_B.cfg.d_model
        # self.cfg = cfg
        # self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        # self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        # self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        # self.buffer = torch.zeros(
        #     (self.buffer_size, 2, model_A.cfg.d_model),
        #     dtype=torch.bfloat16,
        #     requires_grad=False,
        # ).to(cfg["device"]) # hardcoding 2 for model diffing
        self.dA = cfg["d_in_A"]
        self.dB = cfg["d_in_B"]
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A, hook_point=self.cfg["hook_point_A"])
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B, hook_point=self.cfg["hook_point_B"])
        
        self.normalisation_factor = torch.tensor(
        [
            estimated_norm_scaling_factor_A,
            estimated_norm_scaling_factor_B,
        ],
        device="cuda:0",
        dtype=torch.float32,
        )
        self.refresh()
    """
    @torch.no_grad()
    def estimate_norm_scaling_factor_old(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
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
    """

    @torch.no_grad()
    def estimate_norm_scaling_factor(
        self,
        batch_size: int,
        model,
        hook_point: str,
        n_batches_for_norm_estimate: int = 100,
    ):
        norms = []
        seq_len   = self.cfg["seq_len"]
        device    = self.cfg["device"]
        all_toks  = self.all_tokens
        n_tokens  = all_toks.size(0)

        pbar = tqdm.trange(n_batches_for_norm_estimate, desc="Estimating norm scaling factor")
        for _ in pbar:
            # sample batch_size start positions where [s : s+seq_len] is in-bounds
            starts = torch.randint(0, n_tokens - seq_len + 1, (batch_size,), device=device)
            # build a [batch_size, seq_len] LongTensor
            contexts = torch.stack(
                [ all_toks[s : s + seq_len] for s in starts.tolist() ],
                dim=0
            ).to(device=device, dtype=torch.long)

            # this is now a proper [batch, seq_len] input
            _, cache = model.run_with_cache(
                contexts,
                names_filter=hook_point, #either of model A or B
                return_type=None,
            )
            acts = cache[hook_point]  # [batch, seq_len, d_model]
            # mean L2 norm per token position, then average over batch & pos
            norms.append(acts.norm(dim=-1).mean().item())

        mean_norm = float(np.mean(norms))
        return np.sqrt(model.cfg.d_model) / mean_norm



    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        # print("Refreshing the buffer!")

        bs      = self.cfg["model_batch_size"]
        seq_len = self.cfg["seq_len"]
        device  = self.cfg["device"]

        # instead of bs*buffer_mult, need exactly buffer_batches windows
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

        # inputs are first seq_len tokens
        tokens_in = contexts[:, :-1]  # [total_ctx, seq_len]

        # acts_chunks = []
        acts_chunks_A = []
        acts_chunks_B = []

        with torch.autocast(device, torch.bfloat16):
            # process in batches of size bs
            for i in range(0, total_ctx, bs):
                batch = tokens_in[i : i + bs]  # [bs, seq_len]
                _, cacheA = self.model_A.run_with_cache(batch, names_filter=self.cfg["hook_point_A"], return_type=None)
                _, cacheB = self.model_B.run_with_cache(batch, names_filter=self.cfg["hook_point_B"], return_type=None)
                # aA = cacheA[self.cfg["hook_point"]]  # [bs, seq_len, d_model]
                # aB = cacheB[self.cfg["hook_point"]]  # [bs, seq_len, d_model]
                aA = cacheA[self.cfg["hook_point_A"]][:, 1:]  # [bs, seq_len-1, dA]
                aB = cacheB[self.cfg["hook_point_B"]][:, 1:]  # [bs, seq_len-1, dB]

                # drop the BOS token
                # acts = torch.stack([aA, aB], dim=0)[:, :, 1:, :]   # [2, bs, seq_len-1, d_model]
                # flatten bs and seq dimensions
                # acts = einops.rearrange(acts, "n b s d -> (b s) n d")  # [(bs*(seq_len-1)), 2, d_model]
                # acts_chunks.append(acts)
                flatA = einops.rearrange(aA, "b s d -> (b s) d")  # [(bs*(seq_len-1)), dA]
                flatB = einops.rearrange(aB, "b s d -> (b s) d")  # [(bs*(seq_len-1)), dB]
                acts_chunks_A.append(flatA)
                acts_chunks_B.append(flatB)

        fullA = torch.cat(acts_chunks_A, dim=0).to(torch.bfloat16)
        fullB = torch.cat(acts_chunks_B, dim=0).to(torch.bfloat16)


        # full_acts = torch.cat(acts_chunks, dim=0).to(torch.bfloat16)
        # now full_acts.shape == (buffer_batches*(seq_len-1), 2, d_model)
        # assert full_acts.shape == (self.buffer_size, 2, self.model_A.cfg.d_model), (
        #     f"got {full_acts.shape}, expected ({self.buffer_size},2,{self.model_A.cfg.d_model})"
        # )
        assert fullA.shape == (self.buffer_size, self.dA)
        assert fullB.shape == (self.buffer_size, self.dB)

        # shuffle and reset pointer
        # self.buffer  = full_acts[torch.randperm(self.buffer_size, device=device)]
        # self.pointer = 0
        perm = torch.randperm(self.buffer_size, device=self.cfg["device"])
        self.bufA = fullA[perm]
        self.bufB = fullB[perm]


    @torch.no_grad()
    def next(self):
        # out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # # out: [batch_size, n_layers, d_model]
        # self.pointer += self.cfg["batch_size"]
        # if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
        #     self.refresh()
        start = self.pointer
        end   = start + self.cfg["batch_size"]
        A_batch = self.bufA[start:end].float()  # [B, dA]
        B_batch = self.bufB[start:end].float()  # [B, dB]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer_size // 2:
            self.refresh()
        # if self.normalize:
        #     out = out * self.normalisation_factor[None, :, None]

        # apply your per‐model normalisation
        A_batch = A_batch * self.normalisation_factor[0]
        B_batch = B_batch * self.normalisation_factor[1]

        # now return a tuple instead of stacking
        return A_batch, B_batch

        # return out
