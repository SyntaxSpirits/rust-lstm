#![allow(clippy::type_complexity)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use ndarray::{arr2, Array2};
use rust_lstm::*;

#[test]
fn test_network_forward() {
    let input_size = 2;
    let hidden_size = 3;
    let num_layers = 1;

    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    let input = arr2(&[[1.0], [0.5]]);
    let state = network.zero_state(1);

    let (output, _) = network.forward(&input, &state);

    assert_eq!(output.shape(), &[3, 1]);
}

#[test]
fn seeding_makes_initialisation_and_dropout_reproducible() {
    let run = |s: u64| {
        rust_lstm::seed(s);
        let network = LSTMNetwork::new(2, 4, 2)
            .with_recurrent_dropout(0.3, true)
            .with_zoneout(0.1, 0.1);
        let mut trainer = create_basic_trainer(network, 0.01);
        let xs: Vec<_> = (0..5).map(|t| arr2(&[[t as f64 * 0.1], [0.5]])).collect();
        let ys: Vec<_> = (0..5).map(|_| Array2::from_elem((4, 1), 0.2)).collect();
        trainer.train_sequence(&xs, &ys);
        trainer.network.get_cells()[1].w_hh.clone()
    };
    assert_eq!(run(42), run(42));
    assert_ne!(run(42), run(43));
}
