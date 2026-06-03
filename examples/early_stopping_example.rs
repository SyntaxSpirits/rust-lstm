#![allow(clippy::type_complexity)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use ndarray::{arr2, Array2};
use rust_lstm::{
    create_basic_trainer, EarlyStoppingConfig, EarlyStoppingMetric, LSTMNetwork, TrainingConfig,
};

pub(crate) const DEMO_EPOCHS: usize = 6;
pub(crate) const DEMO_PRINT_EVERY: usize = 1;
pub(crate) const MAX_DEMO_PATIENCE: usize = 4;
pub(crate) const DEMO_TRAIN_SEQUENCES: usize = 8;
pub(crate) const DEMO_VALIDATION_SEQUENCES: usize = 3;
pub(crate) const DEMO_SEQUENCE_LENGTH: usize = 5;

type SequencePair = (Vec<Array2<f64>>, Vec<Array2<f64>>);

fn demo_early_stopping_config(
    patience: usize,
    min_delta: f64,
    restore_best_weights: bool,
    monitor: EarlyStoppingMetric,
) -> EarlyStoppingConfig {
    EarlyStoppingConfig {
        patience,
        min_delta,
        restore_best_weights,
        monitor,
    }
}

pub(crate) fn validation_early_stopping_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_EPOCHS,
        print_every: DEMO_PRINT_EVERY,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(demo_early_stopping_config(
            3,
            1e-4,
            true,
            EarlyStoppingMetric::ValidationLoss,
        )),
    }
}

pub(crate) fn train_loss_early_stopping_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_EPOCHS,
        print_every: DEMO_PRINT_EVERY,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(demo_early_stopping_config(
            3,
            0.1,
            true,
            EarlyStoppingMetric::TrainLoss,
        )),
    }
}

pub(crate) fn no_weight_restoration_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_EPOCHS,
        print_every: DEMO_PRINT_EVERY,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(demo_early_stopping_config(
            3,
            1e-4,
            false,
            EarlyStoppingMetric::ValidationLoss,
        )),
    }
}

pub(crate) fn custom_patience_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_EPOCHS,
        print_every: DEMO_PRINT_EVERY,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(demo_early_stopping_config(
            MAX_DEMO_PATIENCE,
            1e-6,
            true,
            EarlyStoppingMetric::ValidationLoss,
        )),
    }
}

fn main() {
    println!("Early Stopping Demonstration");
    println!("================================\n");

    // Generate synthetic data that will overfit quickly
    let (train_data, val_data) = generate_overfitting_data();

    println!(
        "Generated {} training sequences and {} validation sequences",
        train_data.len(),
        val_data.len()
    );

    // Demonstrate different early stopping configurations
    demonstrate_validation_early_stopping(&train_data, &val_data);
    demonstrate_train_loss_early_stopping(&train_data, &val_data);
    demonstrate_no_weight_restoration(&train_data, &val_data);
    demonstrate_custom_patience(&train_data, &val_data);
}

/// Demonstrate early stopping based on validation loss (most common)
fn demonstrate_validation_early_stopping(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("1. VALIDATION LOSS EARLY STOPPING");
    println!("==================================");

    let network = LSTMNetwork::new(1, 8, 1);

    let training_config = validation_early_stopping_training_config();

    let mut trainer = create_basic_trainer(network, 0.01).with_config(training_config);

    println!("Training with validation loss monitoring (patience=3)...");
    trainer.train(train_data, Some(val_data));

    // Show final metrics
    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!(
            "Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}\n",
            final_metrics.epoch,
            final_metrics.train_loss,
            final_metrics.validation_loss.unwrap_or(0.0)
        );
    }
}

/// Demonstrate early stopping based on training loss
fn demonstrate_train_loss_early_stopping(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("2. TRAINING LOSS EARLY STOPPING");
    println!("===============================");

    let network = LSTMNetwork::new(1, 8, 1);

    let training_config = train_loss_early_stopping_training_config();

    let mut trainer = create_basic_trainer(network, 0.01).with_config(training_config);

    println!("Training with training loss monitoring (patience=3)...");
    trainer.train(train_data, Some(val_data));

    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!(
            "Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}\n",
            final_metrics.epoch,
            final_metrics.train_loss,
            final_metrics.validation_loss.unwrap_or(0.0)
        );
    }
}

/// Demonstrate early stopping without weight restoration
fn demonstrate_no_weight_restoration(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("3. EARLY STOPPING WITHOUT WEIGHT RESTORATION");
    println!("=============================================");

    let network = LSTMNetwork::new(1, 8, 1);

    let training_config = no_weight_restoration_training_config();

    let mut trainer = create_basic_trainer(network, 0.01).with_config(training_config);

    println!("Training without weight restoration...");
    trainer.train(train_data, Some(val_data));

    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!(
            "Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}",
            final_metrics.epoch,
            final_metrics.train_loss,
            final_metrics.validation_loss.unwrap_or(0.0)
        );
        println!("Note: Weights are from the last epoch, not the best epoch\n");
    }
}

/// Demonstrate early stopping with custom patience
fn demonstrate_custom_patience(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("4. EARLY STOPPING WITH BOUNDED HIGHER PATIENCE");
    println!("================================================");

    let network = LSTMNetwork::new(1, 8, 1);

    let training_config = custom_patience_training_config();

    let mut trainer = create_basic_trainer(network, 0.01).with_config(training_config);

    println!("Training with bounded higher patience (patience=4)...");
    trainer.train(train_data, Some(val_data));

    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!(
            "Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}\n",
            final_metrics.epoch,
            final_metrics.train_loss,
            final_metrics.validation_loss.unwrap_or(0.0)
        );
    }
}

/// Generate synthetic data that will cause overfitting
/// This creates a simple pattern that's easy to memorize but doesn't generalize well
pub(crate) fn generate_overfitting_data() -> (Vec<SequencePair>, Vec<SequencePair>) {
    let mut train_data = Vec::new();
    let mut val_data = Vec::new();

    // Create training data - simple sine wave with noise
    for i in 0..DEMO_TRAIN_SEQUENCES {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let phase = i as f64 * 0.1;
        for t in 0..DEMO_SEQUENCE_LENGTH {
            let x = (t as f64 * 0.3 + phase).sin();
            let y = ((t + 1) as f64 * 0.3 + phase).sin(); // Next value

            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }

        train_data.push((inputs, targets));
    }

    // Create validation data - different phase to test generalization
    for i in 0..DEMO_VALIDATION_SEQUENCES {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let phase = (i as f64 + 100.0) * 0.1; // Different phase
        for t in 0..DEMO_SEQUENCE_LENGTH {
            let x = (t as f64 * 0.3 + phase).sin();
            let y = ((t + 1) as f64 * 0.3 + phase).sin();

            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }

        val_data.push((inputs, targets));
    }

    (train_data, val_data)
}
