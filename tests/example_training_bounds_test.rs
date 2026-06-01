#[test]
fn training_example_uses_bounded_demo_epochs() {
    let source = include_str!("../examples/training_example.rs");

    assert!(
        source.contains("epochs: 5"),
        "training_example should keep interactive demo training bounded"
    );
    assert!(
        source.contains("fn demo_training_config() -> TrainingConfig"),
        "training_example should centralize its bounded demo training config"
    );
    assert!(
        source
            .matches(".with_config(demo_training_config())")
            .count()
            >= 2,
        "both optimizer demonstrations should apply the bounded demo config"
    );
}

#[test]
fn stock_prediction_example_uses_bounded_demo_epochs() {
    let source = include_str!("../examples/stock_prediction.rs");

    assert!(
        source.contains("epochs: 2"),
        "stock_prediction should avoid the default 100-epoch training config"
    );
    assert!(
        source.contains("StdRng::seed_from_u64(42)"),
        "stock_prediction should keep synthetic data generation reproducible"
    );
    assert!(
        source.contains("generate_stock_data(80)"),
        "stock_prediction should keep its synthetic dataset bounded for interactive runs"
    );
    assert!(
        source.contains("StockPredictor::new(5, 8)"),
        "stock_prediction should keep sequence length and hidden size bounded for interactive runs"
    );
    assert!(
        source.contains(".with_config(demo_training_config())"),
        "stock_prediction should apply the bounded demo config before training"
    );
}

#[test]
fn batch_processing_example_keeps_scalability_demo_bounded() {
    let source = include_str!("../examples/batch_processing_example.rs");

    for expected in [
        "trainer1.config.epochs = 5",
        "trainer2.config.epochs = 5",
        "trainer3.config.epochs = 5",
    ] {
        assert!(
            source.contains(expected),
            "batch benchmark trainer should keep a short epoch budget: {expected}"
        );
    }

    assert!(
        source.contains("trainer.config.epochs = 3"),
        "scalability loop should keep a short epoch budget"
    );
}
