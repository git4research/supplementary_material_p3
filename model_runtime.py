"""Device/dtype resolution, model loading and decoder-block lookup shared by the steering scripts.

Steering hooks attach to model.model.layers[L - 1]; layer L is the output of decoder block L (1-based).
"""
import os


def pick_device(device="auto"):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device: {device}")
    return device


def resolve_dtype(dtype, device, model_id=""):
    """Map --dtype to a torch dtype; 'auto' uses float32 on CPU and bfloat16 on CUDA when supported."""
    import torch

    if dtype == "auto":
        if device == "cpu":
            return torch.float32
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Otherwise Gemma-2 uses float32 rather than float16.
        if "gemma-2" in model_id.lower():
            return torch.float32
        return torch.float16
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    if dtype == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA GPU does not support bfloat16.")
    return getattr(torch, dtype)


def load_model_and_tokenizer(model_id, device="auto", dtype="auto"):
    """Load an eval-mode causal LM and tokenizer, using EOS as the pad token when none is defined."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    resolved_dtype = resolve_dtype(dtype, device, model_id)
    print(f"Loading {model_id} on {device} ({resolved_dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define a padding or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": resolved_dtype, "low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device == "mps":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError(
            f"Unsupported decoder layout for {type(model).__name__}; "
            "expected model.model.layers (Qwen2, Llama, Gemma2)."
        )
    return layers


def get_layer_module(model, layer_index_1based):
    """Return the decoder block for a 1-based layer index (layer L = output of block L)."""
    layers = decoder_layers(model)
    if not 1 <= layer_index_1based <= len(layers):
        raise IndexError(f"Layer {layer_index_1based} outside 1..{len(layers)}")
    return layers[layer_index_1based - 1]
