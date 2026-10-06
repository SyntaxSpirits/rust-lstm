//! Wall-clock timings that `validation/benchmark_pytorch.py` repeats with PyTorch.
//!
//! cargo run --release --example benchmark > validation/rust_timings.csv

use ndarray::Array2;
use rust_lstm::{LSTMNetwork, LossFunction, MSELoss};
use std::hint::black_box;
use std::time::Instant;

const INPUT: usize = 16;
const SEQ_LEN: usize = 50;

fn median_micros(repeats: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..repeats.div_ceil(10) {
        f();
    }
    let mut samples: Vec<f64> = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    println!("task,hidden,batch,microseconds");
    for &hidden in &[16, 64, 256] {
        let mut net = LSTMNetwork::new(INPUT, hidden, 1);
        net.eval();
        let x = Array2::from_elem((INPUT, 1), 0.1);
        let state = net.zero_state(1);
        let step = median_micros(2000, || {
            black_box(net.forward(black_box(&x), &state));
        });
        println!("inference_step,{hidden},1,{step:.2}");

        for &batch in &[1, 16] {
            let xs: Vec<_> = (0..SEQ_LEN)
                .map(|t| Array2::from_elem((INPUT, batch), (t as f64 * 0.1).sin()))
                .collect();
            let ys: Vec<_> = (0..SEQ_LEN)
                .map(|t| Array2::from_elem((hidden, batch), (t as f64 * 0.1).cos()))
                .collect();
            let repeats = if hidden == 256 { 30 } else { 200 };
            let train = median_micros(repeats, || {
                let (outs, caches) = net.forward_sequence_with_cache(&xs);
                let d: Vec<_> = outs
                    .iter()
                    .zip(&ys)
                    .map(|((h, _), y)| MSELoss.compute_batch_gradient(h, y))
                    .collect();
                black_box(net.backward_sequence(&d, &caches));
            });
            println!("forward_backward_T{SEQ_LEN},{hidden},{batch},{train:.2}");
        }
    }
}
