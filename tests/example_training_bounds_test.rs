#[allow(dead_code)]
#[path = "../examples/training_example.rs"]
mod training_example;

#[allow(dead_code)]
#[path = "../examples/stock_prediction.rs"]
mod stock_prediction;

use std::hint::black_box;

#[test]
fn training_example_applies_bounded_config_to_both_demo_trainers() {
    let sgd_trainer = training_example::demo_sgd_trainer(1, 10, 1);
    let adam_trainer = training_example::demo_adam_trainer(1, 10, 1);

    for epochs in [sgd_trainer.config.epochs, adam_trainer.config.epochs] {
        assert!(
            black_box(epochs) <= 5,
            "training_example should keep both optimizer demos bounded"
        );
    }

    for print_every in [
        sgd_trainer.config.print_every,
        adam_trainer.config.print_every,
    ] {
        assert!(
            black_box(print_every) <= black_box(training_example::DEMO_EPOCHS),
            "training_example should report progress without exceeding the epoch budget"
        );
    }
}

#[test]
fn stock_prediction_applies_bounded_config_to_demo_trainer() {
    let network =
        rust_lstm::models::lstm_network::LSTMNetwork::new(5, stock_prediction::DEMO_HIDDEN_SIZE, 2);
    let trainer = stock_prediction::demo_stock_trainer(network);

    assert!(
        black_box(trainer.config.epochs) <= 2,
        "stock_prediction should avoid the default 100-epoch training config"
    );
    assert!(
        black_box(stock_prediction::DEMO_STOCK_DAYS) <= 80,
        "stock_prediction should keep its synthetic dataset bounded for interactive runs"
    );
    assert!(
        black_box(stock_prediction::DEMO_SEQUENCE_LENGTH) <= 5,
        "stock_prediction should keep sequence length bounded for interactive runs"
    );
    assert!(
        black_box(stock_prediction::DEMO_HIDDEN_SIZE) <= 8,
        "stock_prediction should keep hidden size bounded for interactive runs"
    );
}

#[test]
fn stock_prediction_demo_data_is_reproducible() {
    let first = stock_prediction::demo_stock_closes(black_box(8));
    let second = stock_prediction::demo_stock_closes(black_box(8));

    assert_eq!(first, second, "seeded demo data should be reproducible");
    assert!(
        first.windows(2).any(|window| window[0] != window[1]),
        "demo data should still vary across generated days"
    );
}
