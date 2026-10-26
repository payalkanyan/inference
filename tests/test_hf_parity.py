"""The gate: our forward pass must agree with HuggingFace transformers.

If these tests do not pass, nothing else in the project means anything, so they
are written to fail informatively:

  * ``test_prefill_logits_match`` reports the max absolute logit deviation. A
    large number here means the arithmetic is wrong (a wrong RoPE convention, a
    missing bias, a mis-ordered norm).
  * ``test_greedy_token_identical`` is the actual requirement. It can only fail
    while logits still match if a near-tie in the argmax flipped, so reading the
    two together tells you *which* kind of bug you have.

Both models run float32 on CPU and both recompute the full sequence every step,
so this is an apples-to-apples comparison of our arithmetic against theirs, with
no cache implementation on either side to muddy it. Stage 2 onwards compares the
cached paths against this verified baseline.
"""

from __future__ import annotations

import pytest
import torch

from vox.generate import (
    generate_cached,
    generate_engine,
    generate_naive,
    generate_paged,
)
from vox.hub import resolve_model_dir
from vox.sampler import SamplingParams
from vox.weights import load_model

MAX_NEW_TOKENS = 24

RAW_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
]

CHAT_PROMPTS = [
    "What is the capital of France?",
    "List three prime numbers.",
]


@pytest.fixture(scope="module")
def model_dir():
    return resolve_model_dir()


@pytest.fixture(scope="module")
def tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


@pytest.fixture(scope="module")
def vox_model(model_dir):
    return load_model(model_dir, dtype=torch.float32, device="cpu")


@pytest.fixture(scope="module")
def hf_model(model_dir):
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.float32, attn_implementation="sdpa"
        )
    except TypeError:  # transformers < 5 spelled it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float32, attn_implementation="sdpa"
        )
    model.eval()
    return model


def _encode(tokenizer, text: str, chat: bool) -> list[int]:
    if not chat:
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    # transformers 5 returns a BatchEncoding here; older versions returned a bare list.
    return out["input_ids"] if hasattr(out, "keys") else out


def _hf_rope_theta(hf_config) -> float:
    """Dig rope_theta out of whichever shape the installed transformers uses.

    transformers 5 moved it into a ``rope_parameters`` dict and marked it
    per-layer, so reading ``config.rope_theta`` raises outright.
    """
    params = getattr(hf_config, "rope_parameters", None)
    if params:
        assert params.get("rope_type", "default") == "default", (
            f"checkpoint asks for rope_type={params.get('rope_type')!r}; vox only "
            "implements plain RoPE, so this model would decode incorrectly"
        )
        return float(params["rope_theta"])
    return float(hf_config.rope_theta)


@torch.inference_mode()
def _hf_greedy(hf_model, token_ids: list[int], max_new_tokens: int) -> list[int]:
    """Greedy decode with HF, full recompute, no generation_config involvement."""
    tokens = list(token_ids)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        inputs = torch.tensor([tokens], dtype=torch.long)
        logits = hf_model(input_ids=inputs, use_cache=False).logits[:, -1, :]
        next_id = int(logits.argmax(dim=-1)[0])
        tokens.append(next_id)
        produced.append(next_id)
    return produced


def test_config_matches_hf(model_dir, hf_model, vox_model):
    """Our hand-parsed config agrees with HF's, including the details that bite."""
    hf = hf_model.config
    ours = vox_model.cfg
    assert ours.vocab_size == hf.vocab_size
    assert ours.hidden_size == hf.hidden_size
    assert ours.intermediate_size == hf.intermediate_size
    assert ours.num_hidden_layers == hf.num_hidden_layers
    assert ours.num_attention_heads == hf.num_attention_heads
    assert ours.num_key_value_heads == hf.num_key_value_heads
    assert ours.rms_norm_eps == hf.rms_norm_eps
    assert ours.rope_theta == _hf_rope_theta(hf)
    assert ours.tie_word_embeddings == hf.tie_word_embeddings
    assert ours.head_dim == getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    # GQA is real on this model: 14 query heads over 2 KV heads.
    assert ours.kv_repeat > 1


def test_parameter_count_matches_hf(hf_model, vox_model):
    ours = {name: tuple(p.shape) for name, p in vox_model.named_parameters()}
    theirs = {name: tuple(p.shape) for name, p in hf_model.named_parameters()}
    # Tied lm_head shows up in one model's named_parameters and not the other's
    # depending on version; compare on the shared keys and require full coverage
    # of the checkpoint's real weights.
    ours.pop("lm_head.weight", None)
    theirs.pop("lm_head.weight", None)
    assert ours == theirs


@pytest.mark.parametrize("chat", [False, True])
@torch.inference_mode()
def test_prefill_logits_match(tokenizer, hf_model, vox_model, chat):
    prompts = CHAT_PROMPTS if chat else RAW_PROMPTS
    for text in prompts:
        token_ids = _encode(tokenizer, text, chat)
        inputs = torch.tensor([token_ids], dtype=torch.long)

        ours = vox_model(inputs)
        theirs = hf_model(input_ids=inputs, use_cache=False).logits

        assert ours.shape == theirs.shape
        max_dev = (ours - theirs).abs().max().item()
        assert max_dev < 1e-3, f"max logit deviation {max_dev:.3e} for {text!r}"
        # And the thing that actually decides the next token.
        assert int(ours[0, -1].argmax()) == int(theirs[0, -1].argmax())


#: Every single-sequence generation path, so the HF gate covers all of them as
#: they are added rather than each stage checking itself once and moving on.
BACKENDS = {
    "naive": generate_naive,
    "cached": generate_cached,
    "paged": generate_paged,
    "engine": generate_engine,
}


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("chat", [False, True])
def test_greedy_token_identical(tokenizer, hf_model, vox_model, chat, backend):
    """THE correctness requirement: identical token ids under greedy decoding."""
    prompts = CHAT_PROMPTS if chat else RAW_PROMPTS
    for text in prompts:
        token_ids = _encode(tokenizer, text, chat)

        ours = BACKENDS[backend](
            vox_model,
            token_ids,
            SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS),
        ).token_ids
        theirs = _hf_greedy(hf_model, token_ids, MAX_NEW_TOKENS)

        assert ours == theirs, (
            f"divergence for {text!r} on the {backend!r} path\n"
            f"  vox: {ours}\n  ->   {tokenizer.decode(ours)!r}\n"
            f"  hf : {theirs}\n  ->   {tokenizer.decode(theirs)!r}"
        )


@torch.inference_mode()
def test_explicit_position_ids_match_default(vox_model):
    """Later stages always pass position_ids explicitly; check the two agree now."""
    token_ids = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long)
    implicit = vox_model(token_ids)
    explicit = vox_model(token_ids, position_ids=torch.arange(5).unsqueeze(0))
    assert torch.equal(implicit, explicit)


@torch.inference_mode()
def test_constant_position_offset_leaves_output_unchanged(vox_model):
    """RoPE is translation-invariant, and the whole model inherits that.

    Shifting every position by the same amount preserves all pairwise distances,
    so attention scores -- and therefore the logits -- must come out identical.
    """
    token_ids = torch.tensor([[100, 200, 300]], dtype=torch.long)
    at_zero = vox_model(token_ids, position_ids=torch.arange(3).unsqueeze(0))
    at_offset = vox_model(token_ids, position_ids=(torch.arange(3) + 7).unsqueeze(0))
    assert torch.allclose(at_zero, at_offset, atol=1e-4)


@torch.inference_mode()
def test_position_spacing_changes_output(vox_model):
    """...but changing the *spacing* must change the result.

    This is the real guard against position_ids being silently ignored, which
    would still produce fluent output and still pass a naive smoke test.
    """
    token_ids = torch.tensor([[100, 200, 300]], dtype=torch.long)
    tight = vox_model(token_ids, position_ids=torch.tensor([[0, 1, 2]]))
    spread = vox_model(token_ids, position_ids=torch.tensor([[0, 5, 11]]))
    assert not torch.allclose(tight, spread, atol=1e-3)


def test_greedy_sampling_is_deterministic(vox_model, tokenizer):
    token_ids = _encode(tokenizer, CHAT_PROMPTS[0], chat=True)
    params = SamplingParams(temperature=0.0, max_new_tokens=8)
    first = generate_naive(vox_model, token_ids, params).token_ids
    second = generate_naive(vox_model, token_ids, params).token_ids
    assert first == second
