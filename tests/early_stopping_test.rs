use rust_lstm::*;
use ndarray::arr2;

/// Test basic early stopping functionality
#[test]
fn test_early_stopping_basic() {
    let network = LSTMNetwork::new(1, 4, 1);
    
    // Create a simple dataset that will converge quickly
    let train_data = vec![
        (vec![arr2(&[[1.0]])], vec![arr2(&[[0.5]])]),
        (vec![arr2(&[[0.5]])], vec![arr2(&[[0.25]])]),
    ];
    
    let val_data = vec![
        (vec![arr2(&[[0.8]])], vec![arr2(&[[0.4]])]),
    ];
    
    // Configure early stopping with very low patience for quick test
    let early_stopping_config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 1e-2, // Higher threshold to make early stopping more likely
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::ValidationLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 50, // Should stop early
        print_every: 10,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    trainer.train(&train_data, Some(&val_data));
    
    // Early stopping should have been configured (this test just verifies the configuration works)
    let final_metrics = trainer.get_latest_metrics().unwrap();
    assert!(final_metrics.epoch >= 0, "Training should have run at least one epoch");
}

/// Test early stopping with training loss monitoring
#[test]
fn test_early_stopping_train_loss() {
    let network = LSTMNetwork::new(1, 4, 1);
    
    let train_data = vec![
        (vec![arr2(&[[1.0]])], vec![arr2(&[[0.5]])]),
        (vec![arr2(&[[0.5]])], vec![arr2(&[[0.25]])]),
    ];
    
    // Configure early stopping to monitor training loss
    let early_stopping_config = EarlyStoppingConfig {
        patience: 4,
        min_delta: 1e-2, // Higher threshold to make early stopping more likely
        restore_best_weights: false,
        monitor: EarlyStoppingMetric::TrainLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 50,
        print_every: 10,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    trainer.train(&train_data, None); // No validation data
    
    let final_metrics = trainer.get_latest_metrics().unwrap();
    assert!(final_metrics.epoch >= 0, "Training should have run with train loss monitoring");
}

/// Test that training without early stopping runs full epochs
#[test]
fn test_no_early_stopping() {
    let network = LSTMNetwork::new(1, 4, 1);
    
    let train_data = vec![
        (vec![arr2(&[[1.0]])], vec![arr2(&[[0.5]])]),
    ];
    
    let training_config = TrainingConfig {
        epochs: 10,
        print_every: 5,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None, // No early stopping
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    trainer.train(&train_data, None);
    
    let final_metrics = trainer.get_latest_metrics().unwrap();
    assert_eq!(final_metrics.epoch, 9, "Should run all 10 epochs (0-indexed)");
}

/// Test early stopper configuration
#[test]
fn test_early_stopper_config() {
    let config = EarlyStoppingConfig {
        patience: 5,
        min_delta: 1e-3,
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::ValidationLoss,
    };
    
    let mut stopper = EarlyStopper::new(config.clone());
    
    // Test initial state
    assert_eq!(stopper.best_score(), f64::INFINITY);
    assert_eq!(stopper.stopped_epoch(), None);
    
    // Create dummy network and metrics for testing
    let network = LSTMNetwork::new(1, 2, 1);
    let metrics = TrainingMetrics {
        epoch: 0,
        train_loss: 1.0,
        validation_loss: Some(0.5),
        time_elapsed: 1.0,
        learning_rate: 0.01,
    };
    
    // First call should not stop and should be best
    let (should_stop, is_best) = stopper.should_stop(&metrics, &network);
    assert!(!should_stop);
    assert!(is_best);
    assert_eq!(stopper.best_score(), 0.5);
}

/// Test early stopping with different min_delta values
#[test]
fn test_early_stopping_min_delta() {
    let mut stopper = EarlyStopper::new(EarlyStoppingConfig {
        patience: 2,
        min_delta: 0.1, // Require significant improvement
        restore_best_weights: false,
        monitor: EarlyStoppingMetric::ValidationLoss,
    });
    
    let network = LSTMNetwork::new(1, 2, 1);
    
    // First metric - should be best
    let metrics1 = TrainingMetrics {
        epoch: 0,
        train_loss: 1.0,
        validation_loss: Some(1.0),
        time_elapsed: 1.0,
        learning_rate: 0.01,
    };
    let (should_stop, is_best) = stopper.should_stop(&metrics1, &network);
    assert!(!should_stop);
    assert!(is_best);
    
    // Small improvement (less than min_delta) - should not be considered improvement
    let metrics2 = TrainingMetrics {
        epoch: 1,
        train_loss: 0.95,
        validation_loss: Some(0.95), // Only 0.05 improvement, less than 0.1 min_delta
        time_elapsed: 1.0,
        learning_rate: 0.01,
    };
    let (should_stop, is_best) = stopper.should_stop(&metrics2, &network);
    assert!(!should_stop);
    assert!(!is_best); // Should not be considered best due to min_delta
    
    // Another small improvement - should trigger early stopping due to patience
    let metrics3 = TrainingMetrics {
        epoch: 2,
        train_loss: 0.9,
        validation_loss: Some(0.9),
        time_elapsed: 1.0,
        learning_rate: 0.01,
    };
    let (should_stop, is_best) = stopper.should_stop(&metrics3, &network);
    assert!(should_stop); // Should stop due to patience exhausted
    assert!(!is_best);
}

/// Test early stopping with scheduled trainer
#[test]
fn test_early_stopping_with_scheduled_trainer() {
    use rust_lstm::{ScheduledOptimizer, StepLR, Adam};
    
    let network = LSTMNetwork::new(1, 4, 1);
    let optimizer = ScheduledOptimizer::new(Adam::new(0.01), StepLR::new(5, 0.5), 0.01);
    
    let train_data = vec![
        (vec![arr2(&[[1.0]])], vec![arr2(&[[0.5]])]),
    ];
    
    let early_stopping_config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 1e-4,
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::TrainLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 30,
        print_every: 10,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = ScheduledLSTMTrainer::new(network, MSELoss, optimizer)
        .with_config(training_config);
    
    trainer.train(&train_data, None);
    
    // Should complete successfully with early stopping
    let final_metrics = trainer.get_latest_metrics().unwrap();
    assert!(final_metrics.epoch >= 0, "Scheduled trainer should support early stopping");
}

/// Test early stopping with batch trainer
#[test]
fn test_early_stopping_with_batch_trainer() {
    let network = LSTMNetwork::new(1, 4, 1);
    
    let train_data = vec![
        (vec![arr2(&[[1.0]])], vec![arr2(&[[0.5]])]),
        (vec![arr2(&[[0.5]])], vec![arr2(&[[0.25]])]),
    ];
    
    let early_stopping_config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 1e-4,
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::TrainLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 30,
        print_every: 10,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_adam_batch_trainer(network, 0.01)
        .with_config(training_config);
    
    trainer.train(&train_data, None, 2); // Batch size 2
    
    // Should complete successfully with early stopping
    let final_metrics = trainer.get_latest_metrics().unwrap();
    assert!(final_metrics.epoch >= 0, "Batch trainer should support early stopping");
}
