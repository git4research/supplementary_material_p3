"""Forward hooks for inference-time activation steering (Section 2.5.2).

Each steerer adds alpha * delta to the output of one decoder block, by default only on generated
tokens (the multi-token prompt pass is skipped). ConstantDirectionSteer adds a fixed direction
(global_ridge uses ridge_w); LocalManifoldSteer takes a per-token uphill step on the anchor kNN graph;
RandomDirectionSteer is a random-direction control. Example: python 14_steering_hooks.py [--sweep]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch


from model_config import DEFAULT_OUTPUT_ROOT, MODEL_REGISTRY
from model_runtime import get_layer_module

DEFAULT_MODEL_ID = MODEL_REGISTRY["qwen"].model_id
DEFAULT_LAYER = MODEL_REGISTRY["qwen"].default_layer
DEFAULT_MODEL_KEY = f"{MODEL_REGISTRY['qwen'].key}_L{DEFAULT_LAYER}"


class _SteerBase:
    """Base steerer: a forward hook on one decoder block that adds alpha * delta to its output."""

    def __init__(
        self,
        model,
        layer_index_1based: int,
        alpha: float,
        name: str = "",
        scale_mode: Literal["match_hidden", "none"] = "match_hidden",
        debug: bool = False,
    ):
        self.model = model
        self.alpha = float(alpha)
        self.name = name
        self.layer_index_1based = int(layer_index_1based)
        self.scale_mode = scale_mode
        self.debug = bool(debug)

        self._layer_module = self._get_layer_module()

        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._device = next(self._layer_module.parameters()).device
        self._dtype = next(self._layer_module.parameters()).dtype
        self._n_calls = 0
        self.track_pert = False
        self._pert = []


    def _steering_delta(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the steering vector for this forward call, or None to leave the output unchanged."""
        raise NotImplementedError

    def reset(self):
        pass


    def attach(self):
        if self._handle is not None:
            return self
        self._n_calls = 0
        self._handle = self._layer_module.register_forward_hook(self._make_hook())
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc, tb):
        self.detach()


    def _get_layer_module(self):
        return get_layer_module(self.model, self.layer_index_1based)

    def _to_model(self, v: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(v, dtype=np.float32)).to(
            self._device, dtype=self._dtype
        )

    def _make_hook(self):
        """Build the forward-hook entry point that perturbs the block's hidden-state output."""
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            delta_raw = self._steering_delta(hidden)
            # None (e.g., prompt pass or no uphill neighbor) leaves the block output unchanged.
            if delta_raw is None:
                return output

            delta = delta_raw.to(device=hidden.device, dtype=hidden.dtype)

            # match_hidden: rescale delta to the mean L2 norm of the current hidden states (over batch and
            # positions), so the added vector has norm alpha * mean ||h||; in cached decoding this is per token.
            if self.scale_mode == "match_hidden":
                hidden_norm = hidden.float().norm(dim=-1).mean()
                delta_norm  = delta.float().norm()
                if delta_norm > 1e-8:
                    rescale = (hidden_norm / delta_norm).to(hidden.dtype)
                    scaled_delta = delta * rescale
                else:
                    scaled_delta = delta
                modified = hidden + (self.alpha * scaled_delta)
            else:
                modified = hidden + (self.alpha * delta)

            if self.debug and self._n_calls == 0:
                with torch.no_grad():
                    h_n = hidden.float().norm(dim=-1).mean().item()
                    d_n = delta.float().norm().item()
                    eff = (modified - hidden).float().norm(dim=-1).mean().item()
                    print(
                        f"[{self.__class__.__name__}:{self.name}] "
                        f"first call: hidden_shape={tuple(hidden.shape)}, "
                        f"||hidden|| mean={h_n:.3f}, ||delta||={d_n:.4f}, "
                        f"||eff perturbation||={eff:.4f}  "
                        f"(scale_mode={self.scale_mode}, alpha={self.alpha})"
                    )
            if self.track_pert:
                with torch.no_grad():
                    self._pert.append((modified - hidden).float().norm(dim=-1).mean().item())
            self._n_calls += 1

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook

    def mean_perturbation(self):
        return (sum(self._pert) / len(self._pert)) if self._pert else float("nan")


class RandomDirectionSteer(_SteerBase):
    """Control condition: a seeded random unit direction, scaled like the other steerers."""

    def __init__(self, model, alpha, seed: int = 0, steer_prompt: bool = False,
                 layer_index_1based: int = DEFAULT_LAYER,
                 scale_mode: Literal["match_hidden", "none"] = "match_hidden",
                 debug: bool = False, name: str = "random_steer"):
        super().__init__(model, layer_index_1based, alpha, name=name,
                         scale_mode=scale_mode, debug=debug)
        D = int(model.config.hidden_size)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(D).astype(np.float32)
        v = v / (float(np.linalg.norm(v)) + 1e-12)
        self.seed = seed
        self.steer_prompt = bool(steer_prompt)
        self.v = self._to_model(v)

    def _steering_delta(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if hidden.shape[1] > 1 and not self.steer_prompt:
            return None
        return self.v

    def __repr__(self) -> str:
        return f"RandomDirectionSteer(alpha={self.alpha}, seed={self.seed}, layer={self.layer_index_1based})"


class ConstantDirectionSteer(_SteerBase):
    """Adds a constant direction, loaded by name from steering/directions (e.g. ridge_w) or passed in."""

    def __init__(
        self,
        model,
        alpha: float,
        direction_name: Optional[str] = None,
        direction_vector: Optional[np.ndarray] = None,
        layer_index_1based: int = DEFAULT_LAYER,
        model_key: str = DEFAULT_MODEL_KEY,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        unit_norm: bool = False,
        steer_prompt: bool = False,
        scale_mode: Literal["match_hidden", "none"] = "match_hidden",
        debug: bool = False,
        name: str = "",
    ):
        super().__init__(
            model,
            layer_index_1based,
            alpha,
            name=name or (direction_name or "constant_direction"),
            scale_mode=scale_mode,
            debug=debug,
        )
        self.steer_prompt = bool(steer_prompt)

        if (direction_name is None) == (direction_vector is None):
            raise ValueError("Pass exactly one of direction_name or direction_vector.")

        if direction_vector is not None:
            v = np.asarray(direction_vector, dtype=np.float32)
        else:
            v_path = Path(output_root) / model_key / "steering" / "directions" / f"{direction_name}.npy"
            if not v_path.exists():
                raise FileNotFoundError(f"Direction file not found: {v_path}")
            v = np.load(v_path).astype(np.float32)

        if v.ndim != 1:
            raise ValueError(f"Direction vector must be 1-D, got shape {v.shape}")
        if v.shape[0] != model.config.hidden_size:
            raise ValueError(f"Direction dimension {v.shape[0]} != model hidden size {model.config.hidden_size}")
        if not np.isfinite(v).all():
            raise ValueError("Direction vector contains non-finite values.")

        if unit_norm:
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                v = v / n

        self.direction_name = direction_name
        self.v = self._to_model(v)

    def _steering_delta(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        # A multi-token call is the prompt pass; with KV caching each generated token has seq_len == 1.
        if hidden.shape[1] > 1 and not self.steer_prompt:
            return None
        return self.v

    def __repr__(self) -> str:
        return (
            f"ConstantDirectionSteer(direction={self.direction_name}, "
            f"alpha={self.alpha}, steer_prompt={self.steer_prompt}, "
            f"layer={self.layer_index_1based})"
        )


class LocalManifoldSteer(_SteerBase):
    """Per-token steering along the anchor kNN graph toward higher Ridge-predicted score (local_manifold)."""

    def __init__(
        self,
        model,
        alpha: float,
        layer_index_1based: int = DEFAULT_LAYER,
        model_key: str = DEFAULT_MODEL_KEY,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        knn_dir: str = "knn_k30_euclidean",
        top_k: int = 1,
        tau_outer: float = 0.1,
        tau_inner: float = 0.1,
        lam_sim: float = 0.3,
        lam_delta: float = 0.8,
        lam_jump: float = 1.0,
        eps: float = 1e-6,
        steer_prompt: bool = False,
        scale_mode: Literal["match_hidden", "none"] = "match_hidden",
        debug: bool = False,
        name: str = "local_manifold",
    ):
        super().__init__(
            model,
            layer_index_1based,
            alpha,
            name=name,
            scale_mode=scale_mode,
            debug=debug,
        )

        from scipy import sparse

        root = Path(output_root) / model_key
        E_path  = root / "embeddings.npy"
        kn_path = root / knn_dir / "adjacency_csr.npz"
        w_path  = root / "steering" / "directions" / "ridge_w.npy"
        for p in (E_path, kn_path, w_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing manifold artifact: {p}")

        E       = np.load(E_path).astype(np.float32)
        A       = sparse.load_npz(kn_path).tocsr()
        w_ridge = np.load(w_path).astype(np.float32)

        if E.ndim != 2 or w_ridge.ndim != 1:
            raise ValueError("Expected 2-D anchors and a 1-D ridge vector.")
        if E.shape[1] != model.config.hidden_size:
            raise ValueError(f"Anchor dimension {E.shape[1]} != model hidden size {model.config.hidden_size}")
        if not np.isfinite(E).all() or not np.isfinite(w_ridge).all():
            raise ValueError("Manifold artifacts contain non-finite values.")
        if A.shape != (len(E), len(E)):
            raise ValueError("The graph must be square with one row/column per anchor.")

        if E.shape[1] != w_ridge.shape[0]:
            raise ValueError(
                f"Embedding dim {E.shape[1]} != ridge dim {w_ridge.shape[0]}"
            )
        if A.shape[0] != E.shape[0]:
            raise ValueError(
                f"kNN graph rows {A.shape[0]} != n anchors {E.shape[0]}"
            )

        self.H        = torch.from_numpy(E).to(self._device, dtype=torch.float32)
        self.w_ridge  = torch.from_numpy(w_ridge).to(self._device, dtype=torch.float32)
        self.A_indices = torch.from_numpy(A.indices.astype(np.int64)).to(self._device)
        self.A_indptr  = torch.from_numpy(A.indptr.astype(np.int64)).to(self._device)

        self.N = int(self.H.shape[0])
        self.D = int(self.H.shape[1])

        self.top_k     = int(top_k)
        self.tau_outer = float(tau_outer)
        self.tau_inner = float(tau_inner)
        self.lam_sim   = float(lam_sim)
        self.lam_delta = float(lam_delta)
        self.lam_jump  = float(lam_jump)
        self.eps       = float(eps)
        self.steer_prompt = bool(steer_prompt)

        self._step = 0
        self._n_no_uphill = 0

    def reset(self):
        self._step = 0
        self._n_no_uphill = 0
        self._n_calls = 0


    def _graph_neighbors(self, i: int) -> torch.Tensor:
        start = int(self.A_indptr[i].item())
        end   = int(self.A_indptr[i + 1].item())
        return self.A_indices[start:end]

    def _expected_step_from_anchor(self, i: int) -> Optional[torch.Tensor]:
        """Expected uphill step sum_j P(j|i) (x_j - x_i) over graph neighbors j of anchor i; None if none is uphill."""
        nbrs = self._graph_neighbors(i)
        if nbrs.numel() == 0:
            return None

        x_i      = self.H[i]
        nbr_vecs = self.H[nbrs]

        # Score increment of each neighbor along the Ridge direction; keep only uphill moves (> eps).
        deltas = (nbr_vecs - x_i) @ self.w_ridge
        valid_mask = deltas > self.eps
        if not torch.any(valid_mask):
            return None

        valid_nbrs  = nbr_vecs[valid_mask]
        valid_delta = deltas[valid_mask]
        sim_local   = valid_nbrs @ x_i

        # Utility = similarity to anchor i + score gain - quadratic jump penalty; P(j|i) = softmax(utility / tau_inner).
        utility = (
              self.lam_sim   * sim_local
            + self.lam_delta * valid_delta
            - self.lam_jump  * (valid_delta ** 2)
        )
        logits = utility / self.tau_inner
        logits = logits - logits.max()
        probs  = torch.softmax(logits, dim=0)

        expected_step = (probs.unsqueeze(-1) * (valid_nbrs - x_i)).sum(dim=0)
        return expected_step


    def _steering_delta(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """Map the current hidden state to its nearest anchor(s) and return their weighted expected uphill step."""
        seq_len = hidden.shape[1]

        # The prompt pass (seq_len > 1) is skipped unless steer_prompt. Only the first sequence in the
        # batch (its last position) selects anchors; the resulting delta is added to every row.
        if seq_len > 1:
            if not self.steer_prompt:
                return None
            h_current = hidden[0, -1].float()
        else:
            h_current = hidden[0, 0].float()

        h_norm_val = h_current.norm()
        if h_norm_val < 1e-8:
            return None
        h_norm = h_current / h_norm_val

        # Anchors and h are unit-norm, so this is cosine similarity; weight the top_k nearest by softmax(sim / tau_outer).
        sims = self.H @ h_norm
        top_vals, top_ids = torch.topk(sims, self.top_k)
        anchor_weights_outer = torch.softmax(top_vals / self.tau_outer, dim=0)

        transition_vectors: List[torch.Tensor] = []
        valid_weights: List[float] = []
        for j in range(self.top_k):
            i   = int(top_ids[j].item())
            w_i = float(anchor_weights_outer[j].item())
            v_i = self._expected_step_from_anchor(i)
            if v_i is None:
                continue
            transition_vectors.append(v_i)
            valid_weights.append(w_i)

        self._step += 1
        # No uphill neighbor from any selected anchor: no perturbation is applied at this step.
        if not transition_vectors:
            self._n_no_uphill += 1
            return None

        # Renormalize the outer weights over anchors that produced a step.
        weights_t = torch.tensor(valid_weights, device=self._device, dtype=torch.float32)
        weights_t = weights_t / weights_t.sum()
        stacked   = torch.stack(transition_vectors, dim=0)
        v_path    = (weights_t.unsqueeze(-1) * stacked).sum(dim=0)

        return v_path

    def __repr__(self) -> str:
        return (
            f"LocalManifoldSteer(alpha={self.alpha}, top_k={self.top_k}, "
            f"tau_outer={self.tau_outer}, tau_inner={self.tau_inner}, "
            f"lam=({self.lam_sim},{self.lam_delta},{self.lam_jump}), "
            f"layer={self.layer_index_1based})"
        )


def _smoke_test():
    """Greedy generation for one prompt: no steering, global_ridge and local_manifold on the default Qwen artifacts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = DEFAULT_MODEL_ID
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    print(f"Loading {model_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()

    prompt = "I feel like nothing I do makes a difference. What should I do?"
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    MAX_NEW = 300

    def gen():
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    ALPHA = 0.3

    print("\n--- (A) No steering [method #1] ---")
    print(gen())

    print(f"\n--- (B) global_ridge (ridge regression direction), alpha={ALPHA} [method #2] ---")
    cs = ConstantDirectionSteer(
        model,
        alpha=ALPHA,
        direction_name="ridge_w",
        debug=True,
    ).attach()
    try:
        print(gen())
    finally:
        cs.detach()

    print(f"\n--- (C) LocalManifoldSteer (one-step adaptive), alpha={ALPHA} [method #3] ---")
    lm = LocalManifoldSteer(
        model,
        alpha=ALPHA,
        debug=True,
    ).attach()
    try:
        lm.reset()
        print(gen())
        print(f"  [diagnostic] tokens with no-uphill (steering tapered): "
              f"{lm._n_no_uphill}/{lm._step}")
    finally:
        lm.detach()

    print("\nSmoke test done.")


def _sweep_local_manifold():
    """Diagnostic: compare local_manifold outputs across top_k and lam_sim settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = DEFAULT_MODEL_ID
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    print(f"Loading {model_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()

    prompt = "I feel like nothing I do makes a difference. What should I do?"
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    MAX_NEW = 400
    ALPHA = 0.3

    def gen():
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    configs = [
        ("REFERENCE: top_k=10, lam_sim=0.9 (pre-tuning defaults)", dict(top_k=10, lam_sim=0.9)),
        ("k change:  top_k=1,  lam_sim=0.9",                       dict(top_k=1,  lam_sim=0.9)),
        ("lam sweep: top_k=1,  lam_sim=0.5",                       dict(top_k=1,  lam_sim=0.5)),
        ("lam sweep: top_k=1,  lam_sim=0.3 (current default)",     dict(top_k=1,  lam_sim=0.3)),
        ("lam sweep: top_k=1,  lam_sim=0.1",                       dict(top_k=1,  lam_sim=0.1)),
    ]

    print("\n" + "=" * 88)
    print(f"LocalManifoldSteer sweep  (alpha={ALPHA}, max_new_tokens={MAX_NEW})")
    print(f"Prompt: {prompt!r}")
    print("=" * 88)

    print(f"\n--- BASELINE (no steering) ---")
    print(gen())

    for label, kwargs in configs:
        print(f"\n--- {label} ---")
        lm = LocalManifoldSteer(
            model,
            alpha=ALPHA,
            debug=True,
            **kwargs,
        ).attach()
        try:
            lm.reset()
            print(gen())
            print(f"  [diagnostic] no-uphill tokens: {lm._n_no_uphill}/{lm._step}")
        finally:
            lm.detach()

    print("\n" + "=" * 88)
    print("Sweep done.  Read for register shifts toward Growth/Clarity/Unity")
    print("(more reflective, integrative, less mechanistic-advice-list).")
    print("=" * 88)


def _main():
    import argparse
    p = argparse.ArgumentParser(description="Steering hooks smoke / diagnostic")
    p.add_argument("--sweep", action="store_true",
                   help="Run the LocalManifoldSteer (top_k, lam_sim) diagnostic sweep "
                        "instead of the default smoke test.")
    args = p.parse_args()

    if args.sweep:
        _sweep_local_manifold()
    else:
        _smoke_test()


if __name__ == "__main__":
    _main()
