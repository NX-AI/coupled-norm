
# pip install transformers wandb tiktoken datasets hydra-core torch torchvision

import os
from typing import OrderedDict
import sys
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import math
import time
import random
import wandb
import inspect
from dataclasses import dataclass
from torch.nn import functional as F
import numpy as np
import tiktoken
from hellaswag import render_example, iterate_examples
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose
from pathlib import Path

@dataclass
class TrainConfig:
    seed: int = 512
    max_steps: int = 19073 # 19073 # 5000 # 19073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    warmup_steps: int = 715
    total_batch_size: int = 524288 # 262144 # 524288 # 262144 # 2**19, ~0.5M, in number of tokens
    max_lr: float = 6e-4
    min_lr: float = 0.1 * 6e-4
    B: int = 32
    T: int = 1024 # sequence length

    project_name: str = "coupled-rms"
    model_type: str = "gpt2"
    save_model_wandb: bool = False
    save_model_local: bool = True
    save_model_subfolder: str = "model"
    load_pretrained: bool = False
    checkpoint_path: str | None = None # "/home/martin/space/coupled-rms-norm/nanogpt/model_ref-seed-8_19072.pt" # None

    is_ref: bool = False
    coupled_has_bias: bool = False
    use_stat_mul_bias: bool = False
    normal_before: int = 2
    normal_after: int = 20
    normal_before_step: int = 1000
    fix_bias: float = 1.00
    learned_bias_initial: float = 1.05
    stat_update_factor: float = 0.005

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainConfig)

with initialize(version_base=None):
    cfg: TrainConfig = compose(config_name="base_config", overrides=sys.argv[1:])

fix_per_layer = [1.9606101512908936, 1.0616503953933716, 1.0016118288040161, 1.0992448329925537, 1.0188825130462646, 1.0434235334396362, 1.0728251934051514, 1.0478012561798096, 1.107548713684082, 1.095847249031067, 1.0957201719284058, 1.0121983289718628]

if cfg.is_ref:
    run_name = f"ref-s{cfg.seed}f"
else:
    run_name = f"rms-s{cfg.seed}f-no-0_{cfg.normal_before}-{cfg.normal_after}_{11}-w{cfg.normal_before_step}"

tokenizer = tiktoken.get_encoding("gpt2")

dir = Path(__file__).parent
with open(dir / 'wandb.txt', 'r') as file:
    wandb_key = file.read()

torch._dynamo.config.capture_scalar_outputs = True

is_running_in_debugger = sys.gettrace() is not None

ENABLE_WANDB = not is_running_in_debugger
data_root = dir / "edu_fineweb10B"

use_compile = not is_running_in_debugger

wandb_config = OmegaConf.to_container(cfg, resolve=True)
print(f"Config: {wandb_config}")
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT_MLP = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, weight=None, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight if weight is not None else torch.ones(hidden_size,))
        self.eps = eps

    def forward(self, x, scaler=None, return_scaler=False):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if scaler is None:
            var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
            scaler = torch.rsqrt(var)

        y = (self.weight * x * scaler).to(input_dtype)
        return (y, scaler) if return_scaler else y


def rms(x):
    return (x.detach().to(torch.float32).pow(2).mean(dim=-1) + 1e-6).sqrt().mean().item()

def rms_elementwise(x):
    return (x.detach().to(torch.float32).pow(2).mean(dim=-1) + 1e-6).sqrt()

class Block(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.idx = idx
        self.step = 0
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ln_bias = nn.Parameter(torch.full((1,), fill_value=cfg.learned_bias_initial)) if not cfg.is_ref else None
        self.mlp = MLP(config)
        self.register_buffer('norm_change', torch.full((1,), fill_value=1.0))
        self.log_meta = False

    def forward(self, x):
        x_pre_attn_norm = x
        x_n, ln_scale = self.ln_1(x, return_scaler=True)
        ln_scale_x_attn = ln_scale

        if cfg.is_ref or self.idx <= cfg.normal_before or self.idx >= cfg.normal_after or self.step < cfg.normal_before_step:
            ln_scale = None
        else:
            ln_scale = ln_scale * cfg.fix_bias
            if cfg.use_stat_mul_bias:
                ln_scale = ln_scale * (1 / self.norm_change)
            if cfg.coupled_has_bias:
                ln_scale = ln_scale * self.ln_bias

        x_attn = self.attn(x_n)
        x = x + x_attn

        x_pre_mlp_norm = x
        x_n = self.ln_2(x, scaler=ln_scale)
        x_mlp = self.mlp(x_n)
        x = x + x_mlp
        x_out = x

        if cfg.use_stat_mul_bias:
            self.norm_change = self.get_norm_change(x_pre_attn_norm, x_pre_mlp_norm)

        meta = {}
        if self.log_meta:
            ln_scale_x_mlp = (ln_scale if ln_scale is not None else ln_scale_x_attn)

            meta = {
                "rms_pre_attn_norm": rms(x_pre_attn_norm),
                "rms_pre_mlp_norm": rms(x_pre_mlp_norm),
                "rms_attn": rms(x_attn),
                "rms_mlp": rms(x_mlp),
                "rms_out": rms(x_out),
                "norm_attn": rms(x_pre_attn_norm * ln_scale_x_attn),
                "norm_mlp": rms(x_pre_mlp_norm * ln_scale_x_mlp),
                "norm_change_mov_average": self.norm_change,
                "norm_err": rms(x_pre_mlp_norm * ln_scale_x_mlp) - 1,
                "norm_square_err": rms(rms_elementwise(x_pre_mlp_norm * ln_scale_x_mlp) - 1)
            }

            if self.ln_bias is not None:
                meta["ln_bias"] = self.ln_bias.detach().item()

            meta = {f"l2/{self.idx}/{key}": value for key, value in meta.items()}

        return x, meta

    def get_norm_change(self, x_pre_attn, x_pre_mlp):
        new_norm_change = (rms_elementwise(x_pre_mlp) / rms_elementwise(x_pre_attn)).mean()
        return cfg.stat_update_factor * new_norm_change + (1 - cfg.stat_update_factor) * self.norm_change

# -----------------------------------------------------------------------------


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, idx=i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            if hasattr(module, 'NANOGPT_SCALE_INIT_MLP'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        meta = {}
        for i, block in enumerate(self.transformer.h):
            x, block_meta = block(x)
            meta = meta | block_meta

        # forward the final RMSNorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss, acc = None, None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            acc = (torch.argmax(logits, dim=-1) == targets).sum() / targets.numel()
        return logits, loss, acc, meta

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and RMSNorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# vanilla, non-DDP run
ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.seed)

assert cfg.total_batch_size % (cfg.B * cfg.T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = cfg.total_batch_size // (cfg.B * cfg.T * ddp_world_size)
print(f"total desired batch size: {cfg.total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=cfg.B, T=cfg.T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=cfg.B, T=cfg.T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
if cfg.load_pretrained:
    model = GPT.from_pretrained(cfg.model_type) # or init from OpenAI GPT-2

start_step = 0
if cfg.checkpoint_path is not None:
    loaded = torch.load(cfg.checkpoint_path, weights_only=False)
    start_step = loaded["step"]
    model_checkpoint = loaded["model"]
    model_checkpoint = OrderedDict((k.removeprefix('_orig_mod.'), v) for k, v in model_checkpoint.items())
    model.load_state_dict(model_checkpoint)
    print(f"Load model from {cfg.checkpoint_path}. Continue at step {start_step}.")

model.to(device)
if use_compile:
    model = torch.compile(model)

raw_model = model


# Test if model throws any errors
x, y = val_loader.next_batch()
print(x.shape, y.shape)

x, y = train_loader.next_batch()
x, y = x.to(device), y.to(device)
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss, acc, _ = model(x, y)
print(acc)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_steps:
        return cfg.max_lr * (it+1) / cfg.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.max_steps:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)


wandb.login(key=wandb_key)
model_artifact_name = f"{run_name}-model-{random.randint(0, 1000)}"
run = wandb.init(project=cfg.project_name, name=run_name, config=wandb_config, mode=None if ENABLE_WANDB else "disabled")

for step in range(start_step, cfg.max_steps):
    t0 = time.time()
    last_step = (step == cfg.max_steps - 1)

    for block in model.transformer.h:
        block.step = step

    # once in a while evaluate our validation loss
    if step % 50 == 0 or last_step:
        model.eval()
        val_loader.reset()

        for block in model.transformer.h:
            block.log_meta = True

        with torch.no_grad():
            val_loss_accum, val_acc_accum = 0.0, 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss, acc, meta = model(x, y)
    
                val_loss_accum += (loss / val_loss_steps).detach()
                val_acc_accum += (acc / val_loss_steps).detach()

        for block in model.transformer.h:
            block.log_meta = False

        print(f"validation loss: {val_loss_accum.item():.4f} | acc: {val_acc_accum.item():.2f}")
        run.log({"step": step, "val/loss": val_loss_accum.item(), "val/acc": val_acc_accum.item(), **meta})
        print(meta)

        if step > 0 and (step % 5000 == 0 or last_step):
            # optionally write model checkpoints

            if cfg.save_model_local:
                (dir / cfg.save_model_subfolder).mkdir(parents=True, exist_ok=True)
                checkpoint_path = dir / cfg.save_model_subfolder / f"model_{run_name}_{step:05d}.pt"
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

            if cfg.save_model_wandb:
                artifact = wandb.Artifact(name=model_artifact_name, type="model")
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact)


    # once in a while evaluate hellaswag
    if (step % 500 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _, __, ___ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        acc_norm = num_correct_norm / num_total
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        run.log({"step": step, "val/hellaswag_acc": acc_norm, "val/hellaswag_correct": num_correct_norm})

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 100 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = tokenizer.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _, __, ___ = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]# (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum, acc_accum = 0.0, 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss, acc, _ = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        acc_accum += acc / grad_accum_steps
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if step % 10 == 0:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | acc: {acc_accum.item():.6f}  | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    run.log({"step": step, "loss": loss_accum.item(), "acc": acc_accum.item(), "lr":lr, "norm":norm, "dt":1000*dt, "tokens_per_sec": tokens_per_sec })
