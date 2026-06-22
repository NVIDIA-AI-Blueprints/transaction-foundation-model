# Primer 4: Decoder-Only Transformers (Reading a Model Config Without Fear)

**You know:** neural nets exist; you may have trained an MLP.
**You'll learn:** enough transformer anatomy to read this repo's model config line by line, understand the ~29M parameter budget, and know which knobs matter.

This primer is deliberately *config-first*: the goal isn't to derive attention math, it's to make every line of [`configs/pretrain_financial_decoder.yaml`](../../configs/pretrain_financial_decoder.yaml) meaningful.

## The shape of the machine

A decoder-only transformer is a stack of identical layers between two embedding-ish ends:

```
token ids (batch, 4096)
   │
   ▼
Embedding table          vocab 6251 → vectors of size 512
   │
   ▼
8 × Transformer layer    each: self-attention  +  feed-forward (MLP)
   │                      (causally masked)        (SwiGLU, width 1408)
   ▼
Final norm (RMSNorm)
   │
   ├──────────────► hidden states (batch, 4096, 512)   ← embeddings come from here
   ▼
LM head                  512 → 6251 logits per position ← predictions come from here
```

Each layer does two things, repeated 8 times:

- **Self-attention** — every position builds its updated representation as a learned, weighted mixture of (the representations of) *earlier* positions. This is where "context" happens: the token at position 4,000 can draw on transaction patterns from position 12. The **causal mask** zeroes out attention to future positions — this single constraint is what makes next-token training honest ([Primer 3](03-causal-language-modeling.md)).
- **Feed-forward (MLP)** — per-position nonlinear processing; expand 512 → 1408 → back to 512. No cross-position mixing here; think "feature transformation between rounds of communication."

A useful mental model: **attention rounds are communication; MLPs are computation.** Eight alternating rounds let information propagate and combine: layer 1 might notice "this is a small amount", layer 4 "small amounts at this hour are routine for this card", layer 8 "this whole window looks like normal weekday behavior".

## The config, annotated

From [`configs/pretrain_financial_decoder.yaml`](../../configs/pretrain_financial_decoder.yaml):

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
  config:
    _target_: transformers.LlamaConfig   # architecture blueprint, not Meta's weights
    vocab_size: 6251          # MUST equal the tokenizer's vocab — the hard contract
    hidden_size: 512          # width of every token representation (and of embeddings)
    num_hidden_layers: 8      # depth
    num_attention_heads: 8    # attention runs as 8 parallel 64-dim "heads"
    num_key_value_heads: 2    # GQA: 8 query heads share 2 KV heads (4:1)
    intermediate_size: 1408   # MLP inner width (SwiGLU)
    max_position_embeddings: 8192  # positional capacity (RoPE)
    rope_theta: 500000.0      # RoPE base frequency; high = long-context friendly
    hidden_act: silu          # the activation inside SwiGLU
    rms_norm_eps: 1.0e-5      # RMSNorm stabilizer
    attention_dropout: 0.0    # modern LMs skip dropout; data >> parameters
    tie_word_embeddings: false # input embedding table ≠ output LM head
    bos_token_id: 1
    eos_token_id: 2
    pad_token_id: 0           # must match the tokenizer's special-token ids
```

Line-by-line notes you'll actually use:

- **`LlamaConfig` ≠ Llama weights.** We borrow Meta's architecture *recipe* (the class) at custom dimensions and train **from random initialization**. Any HuggingFace decoder config class (Mistral, Qwen, GPT-2…) could be swapped in here — that's the "architecture-agnostic" claim in the README.
- **`vocab_size: 6251`** must exactly match `FinancialTabularTokenizer`'s output. Change the tokenizer (e.g. `merchant_hash_size`), and this number — and the checkpoint — must change with it.
- **Heads** split attention into parallel subspaces (8 heads × 64 dims = 512), letting different heads specialize (one may track temporal tokens, another merchant transitions — at least in folklore; verifying *what* heads track here is a fun [research idea](../05-research/02-improvement-ideas.md)).
- **GQA (`num_key_value_heads: 2`)** — queries keep 8 heads but keys/values are shared across groups of 4. Cuts the KV-cache memory ~4× at inference for minimal quality loss. At 29M params it's more "modern hygiene" than necessity.
- **RoPE (`rope_theta`, `max_position_embeddings`)** — positions are encoded by *rotating* query/key vectors, not by a learned position table. Practical consequences: the model trained at 4,096 tokens can run at 8,192 with graceful (not perfect) degradation, and there's no position-embedding table eating parameters.
- **SwiGLU + RMSNorm + no dropout** — the standard post-2023 Llama recipe (vs GPT-2's GELU + LayerNorm + dropout). `intermediate_size: 1408` is chosen so the SwiGLU MLP (3 weight matrices) costs about what GPT-2's 2048-wide GELU MLP (2 matrices) would — an apples-to-apples parameter budget, as the config comments note.
- **`tie_word_embeddings: false`** — the input table (6251×512) and output head (512×6251) are separate parameters. Tying them would save ~3.2M params; untied is slightly more expressive. At this scale it's a judgment call either way.

## Where the 29M parameters live

Rough budget (helpful when you resize anything):

| Component | Formula | ≈ Params |
|---|---|---|
| Token embeddings | 6,251 × 512 | 3.2M |
| Attention (per layer) | Q: 512² ; KV: 2×(512×128) ; O: 512² | 0.66M |
| MLP (per layer) | 3 × (512 × 1408) | 2.16M |
| × 8 layers | | 22.6M |
| LM head | 512 × 6,251 | 3.2M |
| **Total** | | **≈ 29M** |

Note the bookends: embeddings + head are ~22% of the model *because* the vocab is small. With GPT-2's 50K vocab, they'd dwarf the transformer itself — the parameter-economics argument for domain tokenization from [Primer 2](02-tokenization-and-vocabularies.md).

## Knobs, ordered by how much they matter here

When you start experimenting ([with Loom, gated and tracked](../06-experimentation/01-loom-workflow.md)):

1. **Data and tokenization** — more sequences, better token design: dominates everything below at this scale.
2. **`hidden_size` / `num_hidden_layers`** — model capacity. Scaling-law intuition: grow model and data together; 29M params ↔ ~263M tokens is already data-rich (cf. Chinchilla's ~20 tokens/param rule of thumb).
3. **Context length (`seq_length` in dataset config)** — longer history per prediction vs quadratic attention cost. The literature reports gains up to 2,048 *transactions* of context (nuFormer) — we use 315.
4. **Heads / GQA ratio / `intermediate_size`** — second-order at this scale; keep the standard shapes unless you have a reason.
5. **`rope_theta`, norms, activation** — leave alone; these are stability-tested defaults.

## Key takeaways

- A decoder is `embeddings → N × (causal attention + MLP) → norm → LM head`; hidden size 512 is the width everywhere, which is why embeddings are 512-d.
- The config borrows Llama's *recipe* at custom scale, trained from scratch; any HF decoder architecture slots in.
- `vocab_size` is a contract with the tokenizer; embeddings+head are a big fraction of small models.
- Capacity knobs (width/depth/context) matter; exotic knobs mostly don't, at 29M params.

**Next:** [Primer 5 — Embeddings](05-embeddings.md): pulling vectors out of the trained machine.
