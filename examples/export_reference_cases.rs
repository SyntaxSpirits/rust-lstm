//! Writes forward outputs and BPTT gradients of randomly initialised networks to a
//! JSON file so that `validation/pytorch_parity.py` can recompute them with PyTorch.
//!
//! cargo run --release --example export_reference_cases -- validation/cases.json

use ndarray::Array2;
use rust_lstm::optimizers::Optimizer;
use rust_lstm::{
    Adam, BiLSTMNetwork, CombineMode, GRUNetwork, LSTMCellGradients, LSTMNetwork, LossFunction,
    MSELoss, RMSprop,
};
use serde_json::{json, Value};

fn matrix(m: &Array2<f64>) -> Value {
    json!(m.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>())
}

fn matrices(ms: &[Array2<f64>]) -> Value {
    Value::Array(ms.iter().map(matrix).collect())
}

fn random_sequence(len: usize, rows: usize, batch: usize) -> Vec<Array2<f64>> {
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    (0..len)
        .map(|_| Array2::random((rows, batch), Uniform::new(-1.0, 1.0)))
        .collect()
}

fn mse_terms(outputs: &[Array2<f64>], targets: &[Array2<f64>]) -> (f64, Vec<Array2<f64>>) {
    let loss = outputs
        .iter()
        .zip(targets)
        .map(|(o, y)| MSELoss.compute_batch_loss(o, y))
        .sum();
    let grads = outputs
        .iter()
        .zip(targets)
        .map(|(o, y)| MSELoss.compute_batch_gradient(o, y))
        .collect();
    (loss, grads)
}

fn lstm_params(cells: &[rust_lstm::LSTMCell]) -> Value {
    Value::Array(
        cells
            .iter()
            .map(|c| {
                json!({
                    "w_ih": matrix(&c.w_ih), "w_hh": matrix(&c.w_hh),
                    "b_ih": matrix(&c.b_ih), "b_hh": matrix(&c.b_hh),
                })
            })
            .collect(),
    )
}

fn lstm_grads(grads: &[LSTMCellGradients]) -> Value {
    Value::Array(
        grads
            .iter()
            .map(|g| {
                json!({
                    "w_ih": matrix(&g.w_ih), "w_hh": matrix(&g.w_hh),
                    "b_ih": matrix(&g.b_ih), "b_hh": matrix(&g.b_hh),
                })
            })
            .collect(),
    )
}

fn lstm_case(input: usize, hidden: usize, layers: usize, len: usize, batch: usize) -> Value {
    let mut net = LSTMNetwork::new(input, hidden, layers);
    net.eval();
    let xs = random_sequence(len, input, batch);
    let ys = random_sequence(len, hidden, batch);
    let (outs, caches) = net.forward_sequence_with_cache(&xs);
    let outputs: Vec<_> = outs.into_iter().map(|(h, _)| h).collect();
    let (loss, d_outputs) = mse_terms(&outputs, &ys);
    let (grads, d_inputs) = net.backward_sequence(&d_outputs, &caches);
    json!({
        "kind": "lstm", "input_size": input, "hidden_size": hidden, "num_layers": layers,
        "inputs": matrices(&xs), "targets": matrices(&ys),
        "params": lstm_params(net.get_cells()),
        "outputs": matrices(&outputs), "loss": loss,
        "grads": lstm_grads(&grads), "d_inputs": matrices(&d_inputs),
    })
}

fn bilstm_case(input: usize, hidden: usize, layers: usize, len: usize, batch: usize) -> Value {
    let mut net = BiLSTMNetwork::new(input, hidden, layers, CombineMode::Concat);
    net.eval();
    let xs = random_sequence(len, input, batch);
    let ys = random_sequence(len, 2 * hidden, batch);
    let (outputs, cache) = net.forward_sequence_with_cache(&xs);
    let (loss, d_outputs) = mse_terms(&outputs, &ys);
    let (f_grads, b_grads, d_inputs) = net.backward_sequence(&d_outputs, &cache);
    json!({
        "kind": "bilstm", "input_size": input, "hidden_size": hidden, "num_layers": layers,
        "inputs": matrices(&xs), "targets": matrices(&ys),
        "params": lstm_params(net.get_forward_cells()),
        "params_reverse": lstm_params(net.get_backward_cells()),
        "outputs": matrices(&outputs), "loss": loss,
        "grads": lstm_grads(&f_grads), "grads_reverse": lstm_grads(&b_grads),
        "d_inputs": matrices(&d_inputs),
    })
}

fn gru_case(input: usize, hidden: usize, layers: usize, len: usize, batch: usize) -> Value {
    let mut net = GRUNetwork::new(input, hidden, layers);
    net.eval();
    let xs = random_sequence(len, input, batch);
    let ys = random_sequence(len, hidden, batch);
    let (outs, caches) = net.forward_sequence_with_cache(&xs);
    let outputs: Vec<_> = outs.into_iter().map(|(h, _)| h).collect();
    let (loss, d_outputs) = mse_terms(&outputs, &ys);
    let (grads, d_inputs) = net.backward_sequence(&d_outputs, &caches);
    let names = [
        "w_ir", "w_hr", "b_ir", "b_hr", "w_iz", "w_hz", "b_iz", "b_hz", "w_ih", "w_hh", "b_ih",
        "b_hh",
    ];
    let params = net
        .get_cells()
        .iter()
        .map(|c| {
            let ms = [
                &c.w_ir, &c.w_hr, &c.b_ir, &c.b_hr, &c.w_iz, &c.w_hz, &c.b_iz, &c.b_hz, &c.w_ih,
                &c.w_hh, &c.b_ih, &c.b_hh,
            ];
            Value::Object(
                names
                    .iter()
                    .zip(ms)
                    .map(|(n, m)| (n.to_string(), matrix(m)))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let grads = grads
        .iter()
        .map(|g| {
            let ms = [
                &g.w_ir, &g.w_hr, &g.b_ir, &g.b_hr, &g.w_iz, &g.w_hz, &g.b_iz, &g.b_hz, &g.w_ih,
                &g.w_hh, &g.b_ih, &g.b_hh,
            ];
            Value::Object(
                names
                    .iter()
                    .zip(ms)
                    .map(|(n, m)| (n.to_string(), matrix(m)))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    json!({
        "kind": "gru", "input_size": input, "hidden_size": hidden, "num_layers": layers,
        "inputs": matrices(&xs), "targets": matrices(&ys),
        "params": params, "outputs": matrices(&outputs), "loss": loss,
        "grads": grads, "d_inputs": matrices(&d_inputs),
    })
}

/// Several parameter tensors updated in a fixed order for a few steps, as a model does.
fn optimizer_case(kind: &str, mut optimizer: impl Optimizer, lr: f64) -> Value {
    let shapes = [(4, 3), (4, 4), (4, 1)];
    let mut params: Vec<Array2<f64>> = shapes
        .iter()
        .map(|&(r, c)| random_sequence(1, r, c).remove(0))
        .collect();
    let initial = params.clone();
    let steps: Vec<Vec<Array2<f64>>> = (0..6)
        .map(|_| {
            shapes
                .iter()
                .map(|&(r, c)| random_sequence(1, r, c).remove(0))
                .collect()
        })
        .collect();
    for gradients in &steps {
        for (i, (p, g)) in params.iter_mut().zip(gradients).enumerate() {
            optimizer.update(&format!("p{i}"), p, g);
        }
    }
    json!({
        "kind": kind, "lr": lr,
        "initial": Value::Array(initial.iter().map(matrix).collect()),
        "gradients": Value::Array(steps.iter().map(|s| matrices(s)).collect()),
        "final": Value::Array(params.iter().map(matrix).collect()),
    })
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/cases.json".to_string());
    let mut cases = Vec::new();
    for &(input, hidden, layers, len, batch) in &[
        (3, 5, 1, 8, 1),
        (4, 8, 2, 20, 3),
        (6, 16, 3, 50, 4),
        (1, 32, 1, 200, 2),
    ] {
        cases.push(lstm_case(input, hidden, layers, len, batch));
        cases.push(gru_case(input, hidden, layers, len, batch));
    }
    cases.push(bilstm_case(3, 5, 1, 10, 2));
    cases.push(bilstm_case(4, 6, 2, 15, 3));
    cases.push(optimizer_case("adam", Adam::new(0.01), 0.01));
    cases.push(optimizer_case("rmsprop", RMSprop::new(0.01), 0.01));

    std::fs::write(&path, serde_json::to_string(&cases).unwrap()).unwrap();
    println!("wrote {} cases to {}", cases.len(), path);
}
