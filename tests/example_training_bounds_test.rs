#[allow(dead_code)]
#[path = "../examples/training_example.rs"]
mod training_example;

#[allow(dead_code)]
#[path = "../examples/stock_prediction.rs"]
mod stock_prediction;

#[allow(dead_code)]
#[path = "../examples/early_stopping_example.rs"]
mod early_stopping_example;

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

#[test]
fn early_stopping_example_applies_bounded_configs_to_all_demo_trainers() {
    let configs = [
        early_stopping_example::validation_early_stopping_training_config(),
        early_stopping_example::train_loss_early_stopping_training_config(),
        early_stopping_example::no_weight_restoration_training_config(),
        early_stopping_example::custom_patience_training_config(),
    ];

    for config in configs {
        assert!(
            black_box(config.epochs) <= 6,
            "early_stopping_example should keep every demo training path bounded"
        );
        assert!(
            black_box(config.print_every) <= black_box(config.epochs),
            "early_stopping_example progress logging should not exceed the epoch budget"
        );
        let early_stopping = config
            .early_stopping
            .expect("early_stopping_example demos should enable early stopping");
        assert!(
            black_box(early_stopping.patience) <= 4,
            "early_stopping_example patience should stay bounded for deterministic CI runs"
        );
    }
}

#[test]
fn early_stopping_example_uses_small_deterministic_fixture() {
    let (first_train, first_val) = early_stopping_example::generate_overfitting_data();
    let (second_train, second_val) = early_stopping_example::generate_overfitting_data();

    assert_eq!(
        first_train, second_train,
        "demo training fixture should be deterministic"
    );
    assert_eq!(
        first_val, second_val,
        "demo validation fixture should be deterministic"
    );
    assert!(
        black_box(first_train.len()) <= 8,
        "early_stopping_example should keep training sequence count bounded"
    );
    assert!(
        black_box(first_val.len()) <= 3,
        "early_stopping_example should keep validation sequence count bounded"
    );
    assert!(
        first_train
            .iter()
            .chain(first_val.iter())
            .all(|(inputs, targets)| inputs.len() <= 5 && targets.len() <= 5),
        "early_stopping_example should keep each fixture sequence bounded"
    );
}
