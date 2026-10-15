"""Candidate-layer grids for layer selection, defined by relative depth (Section 2.5.1 / Appendix M).

Used by 11_get_llm_activations.py and run_steering_pipeline.py. By default the candidates are the blocks at
10%, 20%, ..., 100% of model depth (1-based, rounded half up and deduplicated); a refinement sweep adds blocks
5 percentage points either side of a coarse sweep's chosen layer.
"""

from decimal import Decimal, ROUND_HALF_UP


DEFAULT_LAYER_PERCENTAGES = tuple(range(10, 101, 10))


def parse_layers(value):
    layers = sorted(set(int(part.strip()) for part in value.split(",")))
    if not layers or any(layer < 1 for layer in layers):
        raise ValueError("Layer indices must be positive 1-based integers.")
    return layers


def parse_percentages(value):
    percentages = sorted(set(float(part.strip()) for part in value.split(",")))
    if not percentages or any(not 0 < percentage <= 100 for percentage in percentages):
        raise ValueError("Layer percentages must be finite numbers in (0, 100].")
    return percentages


def add_layer_sweep_arguments(parser, *, refinement_source=False):
    """Add the mutually exclusive --layers / --layer-percentages (/ --refine-from) options."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--layers", type=parse_layers,
                       help="Explicit comma-separated 1-based candidate blocks")
    group.add_argument("--layer-percentages", type=parse_percentages,
                       help="Candidate depths in percent; default: 10,20,...,100")
    if refinement_source:
        group.add_argument("--refine-from",
                           help="Coarse chosen_layer.json; add candidates at its best depth +/-5 percentage points")


def round_layer_position(position, n_hidden_layers):
    return max(1, min(n_hidden_layers, int(position.to_integral_value(rounding=ROUND_HALF_UP))))


def percentage_to_layer(percentage, n_hidden_layers):
    return round_layer_position(Decimal(str(percentage)) * n_hidden_layers / 100, n_hidden_layers)


def make_layer_sweep(n_hidden_layers, *, layers=None, percentages=None):
    """Map requested depths (or explicit layers) to unique 1-based block indices, with provenance."""
    if not isinstance(n_hidden_layers, int) or n_hidden_layers < 1:
        raise ValueError("The model must have at least one transformer block.")
    if layers is not None and percentages is not None:
        raise ValueError("Specify layer indices or percentages, not both.")
    strategy = "explicit_layers" if layers is not None else (
        "uniform_percentages" if percentages is None else "custom_percentages"
    )
    requested = [] if layers is not None else list(
        DEFAULT_LAYER_PERCENTAGES if percentages is None else percentages
    )
    if layers is None and (not requested or any(not 0 < value <= 100 for value in requested)):
        raise ValueError("Layer percentages must be finite numbers in (0, 100].")
    mapping = [{"requested_percentage": float(value),
                "layer": percentage_to_layer(value, n_hidden_layers)} for value in requested]
    selected = sorted(set(layers if layers is not None else [entry["layer"] for entry in mapping]))
    if not selected or any(not isinstance(layer, int) or not 1 <= layer <= n_hidden_layers for layer in selected):
        raise ValueError(f"Requested layers must be in 1..{n_hidden_layers}: {selected}")
    return {
        "strategy": strategy,
        "rounding": "nearest_integer_half_up_clamped_to_1_through_N",
        "requested_percentages": requested,
        "percentage_to_layer": mapping,
        "layers": selected,
        "actual_percentages": {str(layer): 100 * layer / n_hidden_layers for layer in selected},
        "n_unique_layers": len(selected),
    }


def make_refinement_sweep(n_hidden_layers, coarse_selection):
    """Coarse layers plus the blocks at the coarse best layer's depth +/- 5 percentage points."""
    if coarse_selection.get("n_hidden_layers") != n_hidden_layers:
        raise ValueError("Coarse selection and current model have different layer counts.")
    if coarse_selection.get("layer_sweep", {}).get("strategy") == "local_refinement":
        raise ValueError("Refinement must start from a coarse sweep, not another refinement.")
    coarse_layers = make_layer_sweep(
        n_hidden_layers, layers=coarse_selection["all_layers_evaluated"]
    )["layers"]
    center_layer = coarse_selection["recommended_layer"]
    if center_layer not in coarse_layers:
        raise ValueError("The coarse recommended layer is not among its evaluated layers.")
    center_percentage = 100 * center_layer / n_hidden_layers
    mapping = []
    for offset in (-5, 5):
        position = Decimal(center_layer) + Decimal(offset) * n_hidden_layers / 100
        if 0 < position <= n_hidden_layers:
            mapping.append({"requested_percentage": float(position * 100 / n_hidden_layers),
                            "layer": round_layer_position(position, n_hidden_layers)})
    neighbor_layers = [entry["layer"] for entry in mapping]
    combined = make_layer_sweep(n_hidden_layers, layers=coarse_layers + neighbor_layers)
    combined.update({
        "strategy": "local_refinement",
        "requested_percentages": [entry["requested_percentage"] for entry in mapping],
        "percentage_to_layer": mapping,
        "center_layer": center_layer,
        "center_actual_percentage": center_percentage,
        "offset_percentage_points": 5,
        "coarse_layers": coarse_layers,
        "new_layers": sorted(set(neighbor_layers) - set(coarse_layers)),
    })
    return combined
