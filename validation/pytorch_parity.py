"""Recompute the cases written by `examples/export_reference_cases.rs` with PyTorch.

LSTM and bidirectional LSTM cases load the exported weights into torch.nn.LSTM.
torch.nn.GRU applies the reset gate after the recurrent matrix product, whereas
rust-lstm follows Cho et al. (2014) and applies it before, so GRU cases are
recomputed with the same equations written in PyTorch and differentiated by autograd.

    cargo run --release --example export_reference_cases -- validation/cases.json
    python validation/pytorch_parity.py validation/cases.json
"""

import json
import sys

import torch

torch.set_default_dtype(torch.float64)


def tensor(m):
    return torch.tensor(m)


def sequence(ms):
    """List of (features, batch) matrices -> (time, batch, features) tensor."""
    return torch.stack([tensor(m).T for m in ms])


def max_abs(a, b):
    return (a - b).abs().max().item()


def lstm_module(case, bidirectional):
    module = torch.nn.LSTM(
        case["input_size"],
        case["hidden_size"],
        case["num_layers"],
        bidirectional=bidirectional,
    )
    directions = [("", case["params"])]
    if bidirectional:
        directions.append(("_reverse", case["params_reverse"]))
    with torch.no_grad():
        for suffix, params in directions:
            for layer, p in enumerate(params):
                for name in ("w_ih", "w_hh", "b_ih", "b_hh"):
                    kind, part = name.split("_")
                    torch_name = (
                        f"{'weight' if kind == 'w' else 'bias'}_{part}_l{layer}{suffix}"
                    )
                    value = tensor(p[name])
                    getattr(module, torch_name).copy_(
                        value if kind == "w" else value[:, 0]
                    )
    return module


def gru_forward(params, xs):
    hidden = len(params[0]["b_ir"])
    batch = xs.shape[1]
    weights = [{k: tensor(v).requires_grad_() for k, v in p.items()} for p in params]
    states = [torch.zeros(hidden, batch) for _ in params]
    outputs = []
    for x in xs:
        layer_input = x.T
        for layer, w in enumerate(weights):
            h = states[layer]
            r = torch.sigmoid(w["w_ir"] @ layer_input + w["b_ir"] + w["w_hr"] @ h + w["b_hr"])
            z = torch.sigmoid(w["w_iz"] @ layer_input + w["b_iz"] + w["w_hz"] @ h + w["b_hz"])
            n = torch.tanh(w["w_ih"] @ layer_input + w["b_ih"] + w["w_hh"] @ (r * h) + w["b_hh"])
            h = (1 - z) * h + z * n
            states[layer] = h
            layer_input = h
        outputs.append(layer_input.T)
    return torch.stack(outputs), weights


def mse(outputs, targets):
    return sum(((o - y) ** 2).mean() for o, y in zip(outputs, targets))


def compare(case):
    xs = sequence(case["inputs"]).requires_grad_()
    ys = sequence(case["targets"])
    errors = {}

    if case["kind"] in ("lstm", "bilstm"):
        bidirectional = case["kind"] == "bilstm"
        module = lstm_module(case, bidirectional)
        outputs, _ = module(xs)
        loss = mse(outputs, ys)
        loss.backward()
        grad_errors = []
        directions = [("", case["grads"])]
        if bidirectional:
            directions.append(("_reverse", case["grads_reverse"]))
        for suffix, grads in directions:
            for layer, g in enumerate(grads):
                for name in ("w_ih", "w_hh", "b_ih", "b_hh"):
                    kind, part = name.split("_")
                    torch_name = (
                        f"{'weight' if kind == 'w' else 'bias'}_{part}_l{layer}{suffix}"
                    )
                    expected = getattr(module, torch_name).grad
                    actual = tensor(g[name])
                    actual = actual if kind == "w" else actual[:, 0]
                    grad_errors.append(max_abs(actual, expected))
    else:
        outputs, weights = gru_forward(case["params"], xs)
        loss = mse(outputs, ys)
        loss.backward()
        grad_errors = [
            max_abs(tensor(g[name]), w[name].grad)
            for g, w in zip(case["grads"], weights)
            for name in g
        ]

    errors["output"] = max_abs(sequence(case["outputs"]), outputs.detach())
    errors["loss"] = abs(case["loss"] - loss.item())
    errors["param_grad"] = max(grad_errors)
    errors["input_grad"] = max_abs(sequence(case["d_inputs"]), xs.grad)
    return errors


def main(path):
    cases = json.load(open(path))
    print(f"PyTorch {torch.__version__}, float64")
    print(f"{'model':7} {'L':>2} {'H':>3} {'T':>4} {'B':>2}  "
          f"{'output':>9} {'loss':>9} {'dL/dθ':>9} {'dL/dx':>9}")
    worst = 0.0
    for case in cases:
        e = compare(case)
        worst = max(worst, *e.values())
        print(
            f"{case['kind']:7} {case['num_layers']:>2} {case['hidden_size']:>3} "
            f"{len(case['inputs']):>4} {len(case['inputs'][0][0]):>2}  "
            f"{e['output']:9.1e} {e['loss']:9.1e} {e['param_grad']:9.1e} {e['input_grad']:9.1e}"
        )
    print(f"largest absolute difference: {worst:.1e}")
    return worst


if __name__ == "__main__":
    worst = main(sys.argv[1] if len(sys.argv) > 1 else "validation/cases.json")
    sys.exit(0 if worst < 1e-10 else 1)
