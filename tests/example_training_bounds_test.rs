#[allow(dead_code)]
#[path = "../examples/training_example.rs"]
mod training_example;

#[allow(dead_code)]
#[path = "../examples/stock_prediction.rs"]
mod stock_prediction;

#[allow(dead_code)]
#[path = "../examples/early_stopping_example.rs"]
mod early_stopping_example;

#[allow(dead_code)]
#[path = "../examples/learning_rate_scheduling.rs"]
mod learning_rate_scheduling;

#[allow(dead_code)]
#[path = "../examples/advanced_lr_scheduling.rs"]
mod advanced_lr_scheduling;

#[allow(dead_code)]
#[path = "../examples/dropout_example.rs"]
mod dropout_example;

#[allow(dead_code)]
#[path = "../examples/weather_prediction.rs"]
mod weather_prediction;

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

#[test]
fn learning_rate_scheduling_applies_bounded_configs_to_all_demo_paths() {
    let configs = [
        learning_rate_scheduling::step_lr_training_config(),
        learning_rate_scheduling::one_cycle_training_config(),
        learning_rate_scheduling::cosine_annealing_training_config(),
        learning_rate_scheduling::exponential_decay_training_config(),
        learning_rate_scheduling::reduce_on_plateau_training_config(),
        learning_rate_scheduling::scheduler_comparison_training_config(),
    ];

    for config in configs {
        assert!(
            black_box(config.epochs) <= 5,
            "learning_rate_scheduling should keep every demo training path bounded"
        );
        assert!(
            black_box(config.print_every) > 0,
            "learning_rate_scheduling progress logging should stay enabled"
        );
        assert!(
            black_box(config.print_every) <= black_box(config.epochs),
            "learning_rate_scheduling progress logging should not exceed the epoch budget"
        );
        assert!(
            config.early_stopping.is_none(),
            "learning_rate_scheduling examples should avoid hidden early-stopping work"
        );
    }

    assert!(
        black_box(learning_rate_scheduling::DEMO_HIDDEN_SIZE) <= 4,
        "learning_rate_scheduling should keep demo hidden size bounded"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_COMPARISON_HIDDEN_SIZE) <= 4,
        "learning_rate_scheduling should keep comparison hidden size bounded"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_STEP_PERIOD) > 0,
        "step scheduler period should be non-zero"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_STEP_PERIOD)
            <= black_box(learning_rate_scheduling::DEMO_STEP_EPOCHS),
        "step scheduler period should fit inside the demo epoch budget"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_COSINE_PERIOD) > 0,
        "cosine scheduler period should be non-zero"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_COSINE_PERIOD)
            <= black_box(learning_rate_scheduling::DEMO_COSINE_EPOCHS),
        "cosine scheduler period should fit inside the demo epoch budget"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_PLATEAU_PATIENCE) > 0,
        "plateau patience should be non-zero"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_PLATEAU_PATIENCE)
            <= black_box(learning_rate_scheduling::DEMO_PLATEAU_EPOCHS),
        "plateau patience should fit inside the demo epoch budget"
    );
}

#[test]
fn learning_rate_scheduling_uses_small_deterministic_fixture() {
    let first = learning_rate_scheduling::generate_sine_wave_data(black_box(4), 0.0);
    let second = learning_rate_scheduling::generate_sine_wave_data(black_box(4), 0.0);

    assert_eq!(
        first, second,
        "demo sine-wave fixture should be deterministic"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_TRAIN_SEQUENCES) <= 16,
        "learning_rate_scheduling should keep training sequence count bounded"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_VAL_SEQUENCES) <= 4,
        "learning_rate_scheduling should keep validation sequence count bounded"
    );
    assert!(
        black_box(learning_rate_scheduling::DEMO_SEQUENCE_LENGTH) <= 6,
        "learning_rate_scheduling should keep each fixture sequence bounded"
    );
    assert!(
        first.iter().all(|(inputs, targets)| {
            inputs.len() == learning_rate_scheduling::DEMO_SEQUENCE_LENGTH
                && targets.len() == learning_rate_scheduling::DEMO_SEQUENCE_LENGTH
        }),
        "generated fixtures should use the public demo sequence-length bound"
    );
}

#[test]
fn dropout_example_applies_bounded_training_config() {
    let config = dropout_example::dropout_training_config();

    assert!(
        black_box(config.epochs) <= 4,
        "dropout_example should avoid the default 100-epoch training config"
    );
    assert!(
        black_box(config.print_every) > 0,
        "dropout_example progress logging should stay enabled"
    );
    assert!(
        black_box(config.print_every) <= black_box(config.epochs),
        "dropout_example progress logging should not exceed the epoch budget"
    );
    assert!(
        config.early_stopping.is_none(),
        "dropout_example should avoid hidden early-stopping work"
    );
    assert!(
        black_box(dropout_example::DEMO_HIDDEN_SIZE) <= 4,
        "dropout_example should keep demo hidden size bounded"
    );
}

#[test]
fn dropout_example_uses_small_deterministic_fixture() {
    let first = dropout_example::generate_sine_wave_data(
        black_box(dropout_example::DEMO_TRAIN_SEQUENCES),
        black_box(dropout_example::DEMO_SEQUENCE_LENGTH),
    );
    let second = dropout_example::generate_sine_wave_data(
        black_box(dropout_example::DEMO_TRAIN_SEQUENCES),
        black_box(dropout_example::DEMO_SEQUENCE_LENGTH),
    );

    assert_eq!(
        first, second,
        "dropout demo fixture should be deterministic"
    );
    assert!(
        black_box(dropout_example::DEMO_TRAIN_SEQUENCES) <= 8,
        "dropout_example should keep training sequence count bounded"
    );
    assert!(
        black_box(dropout_example::DEMO_SEQUENCE_LENGTH) <= 4,
        "dropout_example should keep each fixture sequence bounded"
    );
    assert!(
        first.iter().all(|(inputs, targets)| {
            inputs.len() == dropout_example::DEMO_SEQUENCE_LENGTH
                && targets.len() == dropout_example::DEMO_SEQUENCE_LENGTH
        }),
        "generated fixtures should use the public demo sequence-length bound"
    );
}

#[test]
fn advanced_lr_scheduling_applies_bounded_configs_to_all_demo_paths() {
    let configs = [
        advanced_lr_scheduling::polynomial_decay_training_config(),
        advanced_lr_scheduling::cyclical_lr_training_config(),
        advanced_lr_scheduling::warmup_scheduler_training_config(),
        advanced_lr_scheduling::advanced_training_config(),
    ];

    for config in configs {
        assert!(
            black_box(config.epochs) <= 5,
            "advanced_lr_scheduling should keep every demo training path bounded"
        );
        assert!(
            black_box(config.print_every) > 0,
            "advanced_lr_scheduling progress logging should stay enabled"
        );
        assert!(
            black_box(config.print_every) <= black_box(config.epochs),
            "advanced_lr_scheduling progress logging should not exceed the epoch budget"
        );
        assert!(
            config.early_stopping.is_none(),
            "advanced_lr_scheduling examples should avoid hidden early-stopping work"
        );
    }

    assert!(
        black_box(advanced_lr_scheduling::DEMO_HIDDEN_SIZE) <= 4,
        "advanced_lr_scheduling should keep demo hidden size bounded"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_ADVANCED_HIDDEN_SIZE) <= 6,
        "advanced_lr_scheduling should keep advanced demo hidden size bounded"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_POLYNOMIAL_ITERS)
            <= black_box(advanced_lr_scheduling::DEMO_POLYNOMIAL_EPOCHS),
        "polynomial scheduler iterations should fit inside the demo epoch budget"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_CYCLICAL_STEP_SIZE)
            <= black_box(advanced_lr_scheduling::DEMO_CYCLICAL_EPOCHS),
        "cyclical step size should fit inside the demo epoch budget"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_WARMUP_EPOCH_COUNT)
            <= black_box(advanced_lr_scheduling::DEMO_WARMUP_EPOCHS),
        "warmup period should fit inside the demo epoch budget"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_VISUALIZATION_STEP_SIZE)
            <= black_box(advanced_lr_scheduling::DEMO_VISUALIZATION_STEPS),
        "visualization scheduler step size should fit inside the visualization budget"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_VISUALIZATION_STEPS) <= 20,
        "schedule visualization should stay bounded for interactive runs"
    );
}

#[test]
fn advanced_lr_scheduling_uses_small_deterministic_fixture() {
    let first = advanced_lr_scheduling::generate_sine_wave_data(black_box(4), 0.0);
    let second = advanced_lr_scheduling::generate_sine_wave_data(black_box(4), 0.0);

    assert_eq!(
        first, second,
        "advanced LR demo fixture should be deterministic"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_TRAIN_SEQUENCES) <= 12,
        "advanced_lr_scheduling should keep training sequence count bounded"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_VAL_SEQUENCES) <= 4,
        "advanced_lr_scheduling should keep validation sequence count bounded"
    );
    assert!(
        black_box(advanced_lr_scheduling::DEMO_SEQUENCE_LENGTH) <= 6,
        "advanced_lr_scheduling should keep each fixture sequence bounded"
    );
    assert!(
        first.iter().all(|(inputs, targets)| {
            inputs.len() == advanced_lr_scheduling::DEMO_SEQUENCE_LENGTH
                && targets.len() == advanced_lr_scheduling::DEMO_SEQUENCE_LENGTH
        }),
        "generated fixtures should use the public demo sequence-length bound"
    );
}

#[test]
fn weather_prediction_applies_bounded_training_config() {
    let config = weather_prediction::weather_training_config();

    assert!(
        black_box(config.epochs) <= 4,
        "weather_prediction should avoid long default training runs"
    );
    assert!(
        black_box(config.print_every) > 0,
        "weather_prediction progress logging should stay enabled"
    );
    assert!(
        black_box(config.print_every) <= black_box(config.epochs),
        "weather_prediction progress logging should not exceed the epoch budget"
    );
    assert!(
        config.early_stopping.is_none(),
        "weather_prediction should avoid hidden early-stopping work"
    );
    assert!(
        black_box(weather_prediction::DEMO_WEATHER_DAYS) <= 80,
        "weather_prediction should keep its synthetic dataset bounded"
    );
    assert!(
        black_box(weather_prediction::DEMO_SEQUENCE_LENGTH) <= 5,
        "weather_prediction should keep each sequence bounded"
    );
    assert!(
        black_box(weather_prediction::DEMO_HIDDEN_SIZE) <= 8,
        "weather_prediction should keep hidden size bounded"
    );
    assert!(
        black_box(weather_prediction::DEMO_RECENT_DAYS)
            >= black_box(
                weather_prediction::DEMO_SEQUENCE_LENGTH + weather_prediction::DEMO_NUM_PREDICTIONS,
            ),
        "weather_prediction should keep enough recent days for every preview prediction"
    );
    assert!(
        black_box(weather_prediction::DEMO_WEATHER_DAYS)
            >= black_box(weather_prediction::DEMO_RECENT_DAYS),
        "weather_prediction dataset should cover the preview prediction window"
    );
}

#[test]
fn weather_prediction_demo_data_is_reproducible() {
    let first = weather_prediction::generate_weather_data(black_box(12));
    let second = weather_prediction::generate_weather_data(black_box(12));

    assert_eq!(
        first, second,
        "weather demo fixture should be deterministic"
    );
    assert!(
        first.windows(2).any(|window| window[0] != window[1]),
        "weather demo fixture should vary across generated days"
    );
    assert!(
        first.iter().all(|weather| {
            weather.humidity >= 20.0
                && weather.humidity <= 95.0
                && weather.cloud_cover >= 0.0
                && weather.cloud_cover <= 100.0
                && weather.precipitation >= 0.0
        }),
        "generated weather fixture should keep bounded meteorological values"
    );
}
