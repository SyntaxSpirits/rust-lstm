use ndarray::{Array2, arr2};
use rust_lstm::{
    LSTMNetwork, create_basic_trainer, TrainingConfig, EarlyStoppingConfig, EarlyStoppingMetric,
    MSELoss, Adam
};

fn main() {
    println!("Early Stopping Demonstration");
    println!("================================\n");

    // Generate synthetic data that will overfit quickly
    let (train_data, val_data) = generate_overfitting_data();
    
    println!("Generated {} training sequences and {} validation sequences", 
             train_data.len(), val_data.len());
    
    // Demonstrate different early stopping configurations
    demonstrate_validation_early_stopping(&train_data, &val_data);
    demonstrate_train_loss_early_stopping(&train_data, &val_data);
    demonstrate_no_weight_restoration(&train_data, &val_data);
    demonstrate_custom_patience(&train_data, &val_data);
}

/// Demonstrate early stopping based on validation loss (most common)
fn demonstrate_validation_early_stopping(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]
) {
    println!("1. VALIDATION LOSS EARLY STOPPING");
    println!("==================================");
    
    let network = LSTMNetwork::new(1, 8, 1);
    
    // Configure early stopping with default settings (validation loss monitoring)
    let early_stopping_config = EarlyStoppingConfig {
        patience: 5,
        min_delta: 1e-4,
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::ValidationLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 100, // Will likely stop early
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    println!("Training with validation loss monitoring (patience=5)...");
    trainer.train(train_data, Some(val_data));
    
    // Show final metrics
    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!("Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}\n", 
                 final_metrics.epoch, 
                 final_metrics.train_loss, 
                 final_metrics.validation_loss.unwrap_or(0.0));
    }
}

/// Demonstrate early stopping based on training loss
fn demonstrate_train_loss_early_stopping(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]
) {
    println!("2. TRAINING LOSS EARLY STOPPING");
    println!("===============================");
    
    let network = LSTMNetwork::new(1, 8, 1);
    
    // Configure early stopping to monitor training loss
    let early_stopping_config = EarlyStoppingConfig {
        patience: 8,
        min_delta: 1e-5,
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::TrainLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 100,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    println!("Training with training loss monitoring (patience=8)...");
    trainer.train(train_data, Some(val_data));
    
    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!("Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}\n", 
                 final_metrics.epoch, 
                 final_metrics.train_loss, 
                 final_metrics.validation_loss.unwrap_or(0.0));
    }
}

/// Demonstrate early stopping without weight restoration
fn demonstrate_no_weight_restoration(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]
) {
    println!("3. EARLY STOPPING WITHOUT WEIGHT RESTORATION");
    println!("=============================================");
    
    let network = LSTMNetwork::new(1, 8, 1);
    
    // Configure early stopping without restoring best weights
    let early_stopping_config = EarlyStoppingConfig {
        patience: 5,
        min_delta: 1e-4,
        restore_best_weights: false, // Don't restore best weights
        monitor: EarlyStoppingMetric::ValidationLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 100,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    println!("Training without weight restoration...");
    trainer.train(train_data, Some(val_data));
    
    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!("Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}", 
                 final_metrics.epoch, 
                 final_metrics.train_loss, 
                 final_metrics.validation_loss.unwrap_or(0.0));
        println!("Note: Weights are from the last epoch, not the best epoch\n");
    }
}

/// Demonstrate early stopping with custom patience
fn demonstrate_custom_patience(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]
) {
    println!("4. EARLY STOPPING WITH HIGH PATIENCE");
    println!("====================================");
    
    let network = LSTMNetwork::new(1, 8, 1);
    
    // Configure early stopping with higher patience
    let early_stopping_config = EarlyStoppingConfig {
        patience: 15, // More patient
        min_delta: 1e-6, // Smaller improvement threshold
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::ValidationLoss,
    };
    
    let training_config = TrainingConfig {
        epochs: 100,
        print_every: 2,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: Some(early_stopping_config),
    };
    
    let mut trainer = create_basic_trainer(network, 0.01)
        .with_config(training_config);
    
    println!("Training with high patience (patience=15)...");
    trainer.train(train_data, Some(val_data));
    
    if let Some(final_metrics) = trainer.get_latest_metrics() {
        println!("Final epoch: {}, Train loss: {:.6}, Val loss: {:.6}\n", 
                 final_metrics.epoch, 
                 final_metrics.train_loss, 
                 final_metrics.validation_loss.unwrap_or(0.0));
    }
}

/// Generate synthetic data that will cause overfitting
/// This creates a simple pattern that's easy to memorize but doesn't generalize well
fn generate_overfitting_data() -> (Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>, Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>) {
    let mut train_data = Vec::new();
    let mut val_data = Vec::new();
    
    // Create training data - simple sine wave with noise
    for i in 0..20 {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        let phase = i as f64 * 0.1;
        for t in 0..10 {
            let x = (t as f64 * 0.3 + phase).sin();
            let y = ((t + 1) as f64 * 0.3 + phase).sin(); // Next value
            
            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }
        
        train_data.push((inputs, targets));
    }
    
    // Create validation data - different phase to test generalization
    for i in 0..5 {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        let phase = (i as f64 + 100.0) * 0.1; // Different phase
        for t in 0..10 {
            let x = (t as f64 * 0.3 + phase).sin();
            let y = ((t + 1) as f64 * 0.3 + phase).sin();
            
            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }
        
        val_data.push((inputs, targets));
    }
    
    (train_data, val_data)
}
