//! Compares every analytic gradient with a central finite difference.

use ndarray::Array2;
use rust_lstm::training::sequence_gradients;
use rust_lstm::{
    BiLSTMNetwork, CharacterEmbedding, CombineMode, CrossEntropyLoss, GRUNetwork, LSTMNetwork,
    LinearLayer, LossFunction, MSELoss, PeepholeLSTMCell,
};

type GruParam = fn(&mut GRUNetwork, usize) -> &mut Array2<f64>;
type PeepholeParam = fn(&mut PeepholeLSTMCell) -> &mut Array2<f64>;

const EPS: f64 = 1e-5;
const TOLERANCE: f64 = 1e-6;

/// Largest relative error between `analytic` and the finite-difference gradient of
/// `loss` with respect to the matrix selected by `param`.
fn max_relative_error<M: Clone>(
    model: &M,
    loss: &dyn Fn(&M) -> f64,
    param: &dyn Fn(&mut M) -> &mut Array2<f64>,
    analytic: &Array2<f64>,
) -> f64 {
    let mut probe = model.clone();
    assert_eq!(param(&mut probe).dim(), analytic.dim());
    let mut worst: f64 = 0.0;
    for (idx, &a) in analytic.indexed_iter() {
        let mut plus = model.clone();
        param(&mut plus)[idx] += EPS;
        let mut minus = model.clone();
        param(&mut minus)[idx] -= EPS;
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * EPS);
        let scale = (numeric.abs() + a.abs()).max(1e-3);
        worst = worst.max((numeric - a).abs() / scale);
    }
    worst
}

fn assert_close<M: Clone>(
    name: &str,
    model: &M,
    loss: &dyn Fn(&M) -> f64,
    param: &dyn Fn(&mut M) -> &mut Array2<f64>,
    analytic: &Array2<f64>,
) {
    let err = max_relative_error(model, loss, param, analytic);
    println!("{name}: {err:.2e}");
    assert!(err < TOLERANCE, "{name}: relative error {err:.3e}");
}

fn data(rows: usize, cols: usize, seed: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((seed * 31 + i * 7 + j * 13) as f64 * 0.618).sin()
    })
}

fn sequence(len: usize, rows: usize, batch: usize, seed: usize) -> Vec<Array2<f64>> {
    (0..len).map(|t| data(rows, batch, seed + t)).collect()
}

fn lstm_loss(net: &LSTMNetwork, xs: &[Array2<f64>], ys: &[Array2<f64>]) -> f64 {
    let mut net = net.clone();
    let mut state = net.zero_state(xs[0].ncols());
    let mut total = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let (out, next, _) = net.forward_with_cache(x, &state);
        total += MSELoss.compute_batch_loss(&out, y);
        state = next;
    }
    total
}

fn lstm_analytic(
    net: &LSTMNetwork,
    xs: &[Array2<f64>],
    ys: &[Array2<f64>],
) -> (Vec<rust_lstm::LSTMCellGradients>, Vec<Array2<f64>>) {
    let mut run = net.clone();
    let mut state = run.zero_state(xs[0].ncols());
    let mut caches = Vec::new();
    let mut d_outputs = Vec::new();
    for (x, y) in xs.iter().zip(ys) {
        let (out, next, cache) = run.forward_with_cache(x, &state);
        d_outputs.push(MSELoss.compute_batch_gradient(&out, y));
        caches.push(cache);
        state = next;
    }
    net.backward_sequence(&d_outputs, &caches)
}

fn check_lstm(net: &LSTMNetwork, xs: &[Array2<f64>], ys: &[Array2<f64>], label: &str) {
    let (grads, d_inputs) = lstm_analytic(net, xs, ys);
    let loss = |n: &LSTMNetwork| lstm_loss(n, xs, ys);
    for (l, g) in grads.iter().enumerate() {
        assert_close(
            &format!("{label} layer {l} w_ih"),
            net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].w_ih,
            &g.w_ih,
        );
        assert_close(
            &format!("{label} layer {l} w_hh"),
            net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].w_hh,
            &g.w_hh,
        );
        assert_close(
            &format!("{label} layer {l} b_ih"),
            net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].b_ih,
            &g.b_ih,
        );
        assert_close(
            &format!("{label} layer {l} b_hh"),
            net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].b_hh,
            &g.b_hh,
        );
    }
    for t in 0..xs.len() {
        let input_loss = |x: &Array2<f64>| {
            let mut perturbed = xs.to_vec();
            perturbed[t] = x.clone();
            lstm_loss(net, &perturbed, ys)
        };
        assert_close(
            &format!("{label} input {t}"),
            &xs[t],
            &input_loss,
            &|x| x,
            &d_inputs[t],
        );
    }
}

#[test]
fn lstm_network_bptt_matches_finite_differences() {
    for layers in 1..=3 {
        let mut net = LSTMNetwork::new(3, 4, layers);
        net.eval();
        let xs = sequence(6, 3, 2, 1);
        let ys = sequence(6, 4, 2, 50);
        check_lstm(&net, &xs, &ys, &format!("{layers}-layer"));
    }
}

#[test]
fn lstm_network_with_dropout_and_zoneout_matches_finite_differences() {
    let mut net = LSTMNetwork::new(3, 4, 2)
        .with_input_dropout(0.3, true)
        .with_recurrent_dropout(0.3, true)
        .with_zoneout(0.2, 0.1);
    net.train();
    for cell in net.get_cells_mut() {
        cell.zoneout.as_mut().unwrap().eval();
    }
    let xs = sequence(5, 3, 2, 3);
    let ys = sequence(5, 4, 2, 70);
    // Samples the variational masks once; later steps reuse them, so the loss is
    // a deterministic function of the parameters.
    let state = net.zero_state(2);
    net.forward(&xs[0], &state);
    check_lstm(&net, &xs, &ys, "dropout");
}

#[test]
fn lstm_network_with_cell_update_dropout_matches_finite_differences() {
    let mut net = LSTMNetwork::new(3, 4, 2).with_cell_update_dropout(0.4, true);
    net.train();
    let xs = sequence(5, 3, 2, 4);
    let ys = sequence(5, 4, 2, 75);
    let state = net.zero_state(2);
    net.forward(&xs[0], &state);
    check_lstm(&net, &xs, &ys, "cell update dropout");
}

#[test]
fn trainer_gradients_match_finite_differences() {
    let mut net = LSTMNetwork::new(2, 3, 2);
    net.eval();
    let xs = sequence(7, 2, 1, 9);
    let ys = sequence(7, 3, 1, 90);
    let (_, grads) = sequence_gradients(&mut net, &MSELoss, &xs, &ys);
    let loss = |n: &LSTMNetwork| {
        let mut n = n.clone();
        let (outs, _) = n.forward_sequence_with_cache(&xs);
        outs.iter()
            .zip(&ys)
            .map(|((h, _), y)| MSELoss.compute_loss(h, y))
            .sum()
    };
    for (l, g) in grads.iter().enumerate() {
        assert_close(
            &format!("trainer layer {l} w_hh"),
            &net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].w_hh,
            &g.w_hh,
        );
    }
}

fn gru_loss(net: &GRUNetwork, xs: &[Array2<f64>], ys: &[Array2<f64>]) -> f64 {
    let mut net = net.clone();
    let mut h: Vec<Array2<f64>> = (0..net.num_layers)
        .map(|_| Array2::zeros((net.hidden_size, xs[0].ncols())))
        .collect();
    let mut total = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        h = net.forward(x, &h);
        total += MSELoss.compute_batch_loss(h.last().unwrap(), y);
    }
    total
}

#[test]
fn gru_network_bptt_matches_finite_differences() {
    for layers in 1..=2 {
        let mut net = GRUNetwork::new(3, 4, layers)
            .with_input_dropout(0.3, true)
            .with_recurrent_dropout(0.3, true)
            .with_candidate_dropout(0.3, true);
        net.train();
        let xs = sequence(5, 3, 2, 5);
        let ys = sequence(5, 4, 2, 60);
        let h0: Vec<Array2<f64>> = (0..layers).map(|_| Array2::zeros((4, 2))).collect();
        net.forward(&xs[0], &h0);

        let mut run = net.clone();
        let mut h = h0;
        let mut caches = Vec::new();
        let mut d_outputs = Vec::new();
        for (x, y) in xs.iter().zip(&ys) {
            let (next, cache) = run.forward_with_cache(x, &h);
            d_outputs.push(MSELoss.compute_batch_gradient(next.last().unwrap(), y));
            caches.push(cache);
            h = next;
        }
        let (grads, d_inputs) = net.backward_sequence(&d_outputs, &caches);

        let loss = |n: &GRUNetwork| gru_loss(n, &xs, &ys);
        for (l, g) in grads.iter().enumerate() {
            let params: [(&str, &Array2<f64>, GruParam); 6] = [
                ("w_ir", &g.w_ir, |n, l| &mut n.get_cells_mut()[l].w_ir),
                ("w_hr", &g.w_hr, |n, l| &mut n.get_cells_mut()[l].w_hr),
                ("w_iz", &g.w_iz, |n, l| &mut n.get_cells_mut()[l].w_iz),
                ("w_hz", &g.w_hz, |n, l| &mut n.get_cells_mut()[l].w_hz),
                ("w_ih", &g.w_ih, |n, l| &mut n.get_cells_mut()[l].w_ih),
                ("w_hh", &g.w_hh, |n, l| &mut n.get_cells_mut()[l].w_hh),
            ];
            for (name, analytic, select) in params {
                assert_close(
                    &format!("GRU {layers}-layer layer {l} {name}"),
                    &net,
                    &loss,
                    &|n| select(n, l),
                    analytic,
                );
            }
            assert_close(
                &format!("GRU layer {l} b_hh"),
                &net,
                &loss,
                &|n| &mut n.get_cells_mut()[l].b_hh,
                &g.b_hh,
            );
        }
        let input_loss = |x: &Array2<f64>| {
            let mut perturbed = xs.clone();
            perturbed[2] = x.clone();
            gru_loss(&net, &perturbed, &ys)
        };
        assert_close("GRU input", &xs[2], &input_loss, &|x| x, &d_inputs[2]);
    }
}

#[test]
fn gru_network_with_zoneout_matches_finite_differences() {
    let mut net = GRUNetwork::new(3, 4, 2).with_zoneout(0.3);
    net.train();
    for cell in net.get_cells_mut() {
        cell.zoneout.as_mut().unwrap().eval();
    }
    let xs = sequence(5, 3, 2, 6);
    let ys = sequence(5, 4, 2, 65);
    let h0: Vec<Array2<f64>> = (0..2).map(|_| Array2::zeros((4, 2))).collect();
    let mut run = net.clone();
    let mut h = h0;
    let mut caches = Vec::new();
    let mut d_outputs = Vec::new();
    for (x, y) in xs.iter().zip(&ys) {
        let (next, cache) = run.forward_with_cache(x, &h);
        d_outputs.push(MSELoss.compute_batch_gradient(next.last().unwrap(), y));
        caches.push(cache);
        h = next;
    }
    let (grads, d_inputs) = net.backward_sequence(&d_outputs, &caches);
    let loss = |n: &GRUNetwork| gru_loss(n, &xs, &ys);
    for (l, g) in grads.iter().enumerate() {
        assert_close(
            &format!("GRU zoneout layer {l} w_hz"),
            &net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].w_hz,
            &g.w_hz,
        );
        assert_close(
            &format!("GRU zoneout layer {l} w_hh"),
            &net,
            &loss,
            &|n| &mut n.get_cells_mut()[l].w_hh,
            &g.w_hh,
        );
    }
    let input_loss = |x: &Array2<f64>| {
        let mut perturbed = xs.clone();
        perturbed[1] = x.clone();
        gru_loss(&net, &perturbed, &ys)
    };
    assert_close(
        "GRU zoneout input",
        &xs[1],
        &input_loss,
        &|x| x,
        &d_inputs[1],
    );
}

#[test]
fn bilstm_bptt_matches_finite_differences() {
    for mode in [CombineMode::Concat, CombineMode::Sum, CombineMode::Average] {
        let mut net = BiLSTMNetwork::new(3, 4, 2, mode.clone());
        net.eval();
        let out_rows = net.output_size();
        let xs = sequence(4, 3, 2, 11);
        let ys = sequence(4, out_rows, 2, 80);
        let loss = |n: &BiLSTMNetwork| {
            let mut n = n.clone();
            n.forward_sequence(&xs)
                .iter()
                .zip(&ys)
                .map(|(o, y)| MSELoss.compute_batch_loss(o, y))
                .sum()
        };
        let (outs, cache) = net.forward_sequence_with_cache(&xs);
        let d_outputs: Vec<_> = outs
            .iter()
            .zip(&ys)
            .map(|(o, y)| MSELoss.compute_batch_gradient(o, y))
            .collect();
        let (f_grads, b_grads, d_inputs) = net.backward_sequence(&d_outputs, &cache);
        for l in 0..2 {
            assert_close(
                &format!("BiLSTM {mode:?} forward layer {l} w_ih"),
                &net,
                &loss,
                &|n| &mut n.get_forward_cells_mut()[l].w_ih,
                &f_grads[l].w_ih,
            );
            assert_close(
                &format!("BiLSTM {mode:?} backward layer {l} w_hh"),
                &net,
                &loss,
                &|n| &mut n.get_backward_cells_mut()[l].w_hh,
                &b_grads[l].w_hh,
            );
            assert_close(
                &format!("BiLSTM {mode:?} backward layer {l} b_ih"),
                &net,
                &loss,
                &|n| &mut n.get_backward_cells_mut()[l].b_ih,
                &b_grads[l].b_ih,
            );
        }
        let input_loss = |x: &Array2<f64>| {
            let mut perturbed = xs.clone();
            perturbed[1] = x.clone();
            let mut n = net.clone();
            n.forward_sequence(&perturbed)
                .iter()
                .zip(&ys)
                .map(|(o, y)| MSELoss.compute_batch_loss(o, y))
                .sum()
        };
        assert_close(
            &format!("BiLSTM {mode:?} input"),
            &xs[1],
            &input_loss,
            &|x| x,
            &d_inputs[1],
        );
    }
}

#[test]
fn peephole_cell_bptt_matches_finite_differences() {
    let cell = PeepholeLSTMCell::new(3, 4);
    let xs = sequence(5, 3, 2, 13);
    let ys = sequence(5, 4, 2, 40);
    let loss = |c: &PeepholeLSTMCell| {
        let mut h = Array2::zeros((4, 2));
        let mut s = Array2::zeros((4, 2));
        let mut total = 0.0;
        for (x, y) in xs.iter().zip(&ys) {
            let (h_next, s_next) = c.forward(x, &h, &s);
            total += MSELoss.compute_batch_loss(&h_next, y);
            h = h_next;
            s = s_next;
        }
        total
    };

    let mut h = Array2::zeros((4, 2));
    let mut s = Array2::zeros((4, 2));
    let mut caches = Vec::new();
    let mut d_outputs = Vec::new();
    for (x, y) in xs.iter().zip(&ys) {
        let (h_next, s_next, cache) = cell.forward_with_cache(x, &h, &s);
        d_outputs.push(MSELoss.compute_batch_gradient(&h_next, y));
        caches.push(cache);
        h = h_next;
        s = s_next;
    }
    let mut dh_next = Array2::zeros((4, 2));
    let mut dc_next = Array2::zeros((4, 2));
    let mut total = None;
    for t in (0..xs.len()).rev() {
        let (g, _, dh, dc) = cell.backward(&(&d_outputs[t] + &dh_next), &dc_next, &caches[t]);
        total = Some(match total {
            None => g,
            Some(mut acc) => {
                add(&mut acc, &g);
                acc
            }
        });
        dh_next = dh;
        dc_next = dc;
    }
    let g = total.unwrap();
    let params: [(&str, &Array2<f64>, PeepholeParam); 8] = [
        ("w_xi", &g.w_xi, |c| &mut c.w_xi),
        ("w_ci", &g.w_ci, |c| &mut c.w_ci),
        ("w_hf", &g.w_hf, |c| &mut c.w_hf),
        ("w_cf", &g.w_cf, |c| &mut c.w_cf),
        ("w_hc", &g.w_hc, |c| &mut c.w_hc),
        ("b_c", &g.b_c, |c| &mut c.b_c),
        ("w_xo", &g.w_xo, |c| &mut c.w_xo),
        ("w_co", &g.w_co, |c| &mut c.w_co),
    ];
    for (name, analytic, select) in params {
        assert_close(&format!("peephole {name}"), &cell, &loss, &select, analytic);
    }
}

fn add(acc: &mut rust_lstm::PeepholeLSTMCellGradients, g: &rust_lstm::PeepholeLSTMCellGradients) {
    acc.w_xi += &g.w_xi;
    acc.w_hi += &g.w_hi;
    acc.b_i += &g.b_i;
    acc.w_ci += &g.w_ci;
    acc.w_xf += &g.w_xf;
    acc.w_hf += &g.w_hf;
    acc.b_f += &g.b_f;
    acc.w_cf += &g.w_cf;
    acc.w_xc += &g.w_xc;
    acc.w_hc += &g.w_hc;
    acc.b_c += &g.b_c;
    acc.w_xo += &g.w_xo;
    acc.w_ho += &g.w_ho;
    acc.b_o += &g.b_o;
    acc.w_co += &g.w_co;
}

#[derive(Clone)]
struct Classifier {
    embedding: CharacterEmbedding,
    lstm: LSTMNetwork,
    head: LinearLayer,
}

fn classifier_loss(m: &Classifier, tokens: &[usize], labels: &[Array2<f64>]) -> f64 {
    let mut m = m.clone();
    let embedded = m.embedding.forward(tokens);
    let xs: Vec<_> = embedded
        .rows()
        .into_iter()
        .map(|r| r.to_owned().insert_axis(ndarray::Axis(1)))
        .collect();
    let outs = m.lstm.forward_sequence(&xs);
    outs.iter()
        .zip(labels)
        .map(|(h, y)| CrossEntropyLoss.compute_loss(&m.head.forward(h), y))
        .sum()
}

#[test]
fn embedding_lstm_linear_cross_entropy_matches_finite_differences() {
    let mut lstm = LSTMNetwork::new(3, 4, 1);
    lstm.eval();
    let model = Classifier {
        embedding: CharacterEmbedding::new(6, 3),
        lstm,
        head: LinearLayer::new(4, 5),
    };
    let tokens = [1, 4, 2, 2, 5];
    let labels: Vec<Array2<f64>> = tokens
        .iter()
        .map(|&t| Array2::from_shape_fn((5, 1), |(i, _)| if i == (t + 1) % 5 { 1.0 } else { 0.0 }))
        .collect();

    let mut m = model.clone();
    let embedded = m.embedding.forward(&tokens);
    let xs: Vec<_> = embedded
        .rows()
        .into_iter()
        .map(|r| r.to_owned().insert_axis(ndarray::Axis(1)))
        .collect();
    let (outs, caches) = m.lstm.forward_sequence_with_cache(&xs);
    let mut head_weight = Array2::zeros(m.head.weight.raw_dim());
    let mut d_outputs = Vec::new();
    for ((h, _), y) in outs.iter().zip(&labels) {
        let logits = m.head.forward(h);
        let (g, dh) = m
            .head
            .backward(&CrossEntropyLoss.compute_gradient(&logits, y));
        head_weight += &g.weight;
        d_outputs.push(dh);
    }
    let (lstm_grads, d_inputs) = m.lstm.backward_sequence(&d_outputs, &caches);
    let mut d_embedded = Array2::zeros(embedded.raw_dim());
    for (t, d) in d_inputs.iter().enumerate() {
        d_embedded.row_mut(t).assign(&d.column(0));
    }
    let embedding_grad = m.embedding.backward(&d_embedded).weight;

    let loss = |c: &Classifier| classifier_loss(c, &tokens, &labels);
    assert_close(
        "head weight",
        &model,
        &loss,
        &|c| &mut c.head.weight,
        &head_weight,
    );
    assert_close(
        "lstm w_hh",
        &model,
        &loss,
        &|c| &mut c.lstm.get_cells_mut()[0].w_hh,
        &lstm_grads[0].w_hh,
    );
    assert_close(
        "embedding",
        &model,
        &loss,
        &|c| &mut c.embedding.weight,
        &embedding_grad,
    );
}

#[test]
fn padded_batch_matches_individual_sequences() {
    let mut net = LSTMNetwork::new(2, 3, 2);
    net.eval();
    let sequences = vec![
        sequence(5, 2, 1, 1),
        sequence(3, 2, 1, 2),
        sequence(4, 2, 1, 3),
    ];
    let batched = net.forward_batch_sequences(&sequences);
    for (seq, outputs) in sequences.iter().zip(&batched) {
        let single = net.forward_sequence(seq);
        assert_eq!(single.len(), outputs.len());
        for (a, (b, _)) in single.iter().zip(outputs) {
            assert!((a - b).iter().all(|d| d.abs() < 1e-12));
        }
    }
}
