#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformer-lens')


# In[23]:


import torch as t
import numpy as np
import torch.nn as nn
import circuitsvis as cv
from IPython.display import display
from transformer_lens import HookedTransformer
import tests
import math
from jaxtyping import Float, Int
from torch import Tensor
import einops


# In[3]:


reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()


# In[4]:


len(list(reference_gpt2.tokenizer.vocab.items()))


# In[5]:


reference_text = 'Hi! I am learning transformers. AI will take over the world'
tokens = reference_gpt2.to_tokens(reference_text)
print(tokens)


# In[6]:


logits, cache = reference_gpt2.run_with_cache(tokens)


# In[7]:


print(logits)


# In[8]:


print(logits.shape)
print(tokens.shape)


# In[9]:


probs = logits.softmax(dim=-1)
print(probs.shape)


# In[10]:


most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))


# In[11]:


for activation_name, activation in cache.items():
  print(f"{activation_name:30}: {tuple(activation.shape)}")


# In[12]:


for name, param in reference_gpt2.named_parameters():
    if ".0." in name or "blocks" not in name:
        print(f"{name:30}: {tuple(param.shape)}")


# In[13]:


print(reference_gpt2.cfg)


# In[14]:


from dataclasses import dataclass


# In[15]:


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)


# In[16]:


def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct\n")
    assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, "More than 0.01% of the values are incorrect"


# In[17]:


pip install jaxtyping


# In[19]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[20]:


print(device)


# In[21]:


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b


# In[24]:


rand_float_test(LayerNorm, [2,4,768])


# In[25]:


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


# In[26]:


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


# In[27]:


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# In[28]:


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre, self.W_Q, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre, self.W_K, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre, self.W_V, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"
            )
            + self.b_V
        )
        attn_scores = einops.einsum(
            q, k, "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K"
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)
        z = einops.einsum(
            v, attn_pattern, "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head"
        )
        attn_out = (
            einops.einsum(z, self.W_O, "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model")
            + self.b_O
        )

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# In[29]:


rand_float_test(Attention, [2, 4, 768])


# In[30]:


display(
    cv.attention.attention_patterns(
        tokens=reference_gpt2.to_str_tokens(reference_text), attention=cache["pattern", 0][0]
    )
)


# In[31]:


class MLP(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
    self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
    self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
    self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
    nn.init.normal_(self.W_in, std=self.cfg.init_range)
    nn.init.normal_(self.W_out, std=self.cfg.init_range)

  def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    layer_1 = einops.einsum(normalized_resid_mid, self.W_in, "batch posn d_model, d_model d_mlp -> batch posn d_mlp") + self.b_in
    gelu = torch.nn.GELU()
    output1 = gelu(layer_1)
    layer_2 = einops.einsum(output1, self.W_out, "batch posn d_mlp, d_mlp d_model -> batch posn d_model") + self.b_out
    return layer_2


# In[32]:


rand_float_test(MLP, [2,4,768])


# In[33]:


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


# In[34]:


rand_float_test(TransformerBlock, [2, 4, 768])


# In[35]:


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )


# In[36]:


rand_float_test(Unembed, [2, 4, 768])


# In[37]:


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits


# In[38]:


rand_int_test(DemoTransformer, [2, 4])


# In[39]:


demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)


# In[40]:


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs_for_tokens


pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")


# In[41]:


from tqdm import tqdm


# In[42]:


test_string = """earth revolve around """
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)


# In[ ]:




