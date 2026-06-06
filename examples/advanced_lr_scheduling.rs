#![allow(clippy::type_complexity)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use ndarray::{arr2, Array2};
use rust_lstm::{
    Adam, CyclicalLR, CyclicalMode, LRScheduleVisualizer, LSTMNetwork, MSELoss, PolynomialLR,
    ScheduledLSTMTrainer, ScheduledOptimizer, StepLR, TrainingConfig, WarmupScheduler,
};

pub const DEMO_TRAIN_SEQUENCES: usize = 12;
pub const DEMO_VAL_SEQUENCES: usize = 4;
pub const DEMO_SEQUENCE_LENGTH: usize = 6;
pub const DEMO_HIDDEN_SIZE: usize = 4;
pub const DEMO_ADVANCED_HIDDEN_SIZE: usize = 6;
pub const DEMO_POLYNOMIAL_EPOCHS: usize = 5;
pub const DEMO_CYCLICAL_EPOCHS: usize = 5;
pub const DEMO_WARMUP_EPOCHS: usize = 5;
pub const DEMO_ADVANCED_EPOCHS: usize = 5;
pub const DEMO_POLYNOMIAL_ITERS: usize = DEMO_POLYNOMIAL_EPOCHS;
pub const DEMO_CYCLICAL_STEP_SIZE: usize = 2;
pub const DEMO_WARMUP_EPOCH_COUNT: usize = 2;
pub const DEMO_BASE_STEP_SIZE: usize = 2;
pub const DEMO_VISUALIZATION_STEP_SIZE: usize = 2;
pub const DEMO_VISUALIZATION_STEPS: usize = 20;

fn main() {
    println!("🚀 Advanced Learning Rate Scheduling for Rust-LSTM");
    println!("===================================================\n");

    // Generate sample training data
    let train_data = generate_sine_wave_data(DEMO_TRAIN_SEQUENCES, 0.0);
    let val_data = generate_sine_wave_data(DEMO_VAL_SEQUENCES, 1000.0);

    // 1. Polynomial Decay Example
    polynomial_decay_example(&train_data, &val_data);

    // 2. Cyclical Learning Rate Examples
    cyclical_lr_examples(&train_data, &val_data);

    // 3. Warmup Scheduler Example
    warmup_scheduler_example(&train_data, &val_data);

    // 4. Schedule Visualization
    schedule_visualization();

    // 5. Advanced Training with Best Practices
    advanced_training_example(&train_data, &val_data);
}

fn polynomial_decay_example(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("1️⃣  Polynomial Decay Example");
    println!("   Smoothly decays LR using polynomial function\n");

    let network = LSTMNetwork::new(1, DEMO_HIDDEN_SIZE, 1);

    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::polynomial(
        Adam::new(0.01),
        0.01,                  // base_lr
        DEMO_POLYNOMIAL_ITERS, // total_iters
        2.0,                   // power
        0.001,                 // end_lr
    );

    let config = polynomial_decay_training_config();

    let mut trainer =
        ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer).with_config(config);

    trainer.train(train_data, Some(val_data));

    println!("Final LR: {:.2e}\n", trainer.get_current_lr());
    println!("----------------------------------------\n");
}

fn cyclical_lr_examples(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("2️⃣  Cyclical Learning Rate Examples");
    println!("   Oscillates between min and max LR with different patterns\n");

    // 2a. Triangular Cyclical LR
    println!("2a. Triangular Cyclical LR");
    let network = LSTMNetwork::new(1, DEMO_HIDDEN_SIZE, 1);

    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::cyclical(
        Adam::new(0.001),
        0.001,                   // base_lr
        0.01,                    // max_lr
        DEMO_CYCLICAL_STEP_SIZE, // step_size
    );

    let config = cyclical_lr_training_config();

    let mut trainer =
        ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer).with_config(config);

    trainer.train(train_data, Some(val_data));
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());

    // 2b. Triangular2 Cyclical LR (halving amplitude each cycle)
    println!("2b. Triangular2 Cyclical LR (halving amplitude each cycle)");
    let network = LSTMNetwork::new(1, DEMO_HIDDEN_SIZE, 1);

    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::cyclical_triangular2(
        Adam::new(0.001),
        0.001,                   // base_lr
        0.01,                    // max_lr
        DEMO_CYCLICAL_STEP_SIZE, // step_size
    );

    let config2 = cyclical_lr_training_config();

    let mut trainer =
        ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer).with_config(config2);

    trainer.train(train_data, Some(val_data));
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());

    // 2c. ExpRange Cyclical LR (exponential scaling)
    println!("2c. ExpRange Cyclical LR (exponential scaling)");
    let network = LSTMNetwork::new(1, DEMO_HIDDEN_SIZE, 1);

    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::cyclical_exp_range(
        Adam::new(0.001),
        0.001,                   // base_lr
        0.01,                    // max_lr
        DEMO_CYCLICAL_STEP_SIZE, // step_size
        0.95,                    // gamma
    );

    let config3 = cyclical_lr_training_config();

    let mut trainer =
        ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer).with_config(config3);

    trainer.train(train_data, Some(val_data));
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());

    println!("----------------------------------------\n");
}

fn warmup_scheduler_example(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("3️⃣  Warmup Scheduler Example");
    println!("   Gradually increases LR during warmup, then applies base scheduler\n");

    let network = LSTMNetwork::new(1, DEMO_HIDDEN_SIZE, 1);

    let base_scheduler = StepLR::new(DEMO_BASE_STEP_SIZE, 0.5);
    let warmup_scheduler = WarmupScheduler::new(DEMO_WARMUP_EPOCH_COUNT, base_scheduler, 0.001);

    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::new(Adam::new(0.01), warmup_scheduler, 0.01);

    let config = warmup_scheduler_training_config();

    let mut trainer =
        ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer).with_config(config);

    trainer.train(train_data, Some(val_data));

    println!("Final LR: {:.2e}\n", trainer.get_current_lr());
    println!("----------------------------------------\n");
}

fn schedule_visualization() {
    println!("4️⃣  Learning Rate Schedule Visualization");
    println!("   ASCII visualization of different schedulers\n");

    // Visualize StepLR
    println!("StepLR (step_size=2, gamma=0.5):");
    let step_scheduler = StepLR::new(DEMO_VISUALIZATION_STEP_SIZE, 0.5);
    LRScheduleVisualizer::print_schedule(step_scheduler, 0.01, DEMO_VISUALIZATION_STEPS, 40, 5);
    println!();

    // Visualize PolynomialLR
    println!("PolynomialLR (power=2.0, end_lr=0.001):");
    let poly_scheduler = PolynomialLR::new(DEMO_VISUALIZATION_STEPS, 2.0, 0.001);
    LRScheduleVisualizer::print_schedule(poly_scheduler, 0.01, DEMO_VISUALIZATION_STEPS, 40, 5);
    println!();

    // Visualize CyclicalLR
    println!("CyclicalLR Triangular (base_lr=0.001, max_lr=0.01, step_size=2):");
    let cyclical_scheduler = CyclicalLR::new(0.001, 0.01, DEMO_CYCLICAL_STEP_SIZE);
    LRScheduleVisualizer::print_schedule(
        cyclical_scheduler,
        0.001,
        DEMO_VISUALIZATION_STEPS,
        40,
        5,
    );
    println!();

    println!("----------------------------------------\n");
}

fn advanced_training_example(
    train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
    val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)],
) {
    println!("5️⃣  Advanced Training with Best Practices");
    println!("   Warmup + Cyclical LR + Dropout + Gradient Clipping\n");

    // Create network with dropout
    let network = LSTMNetwork::new(1, DEMO_ADVANCED_HIDDEN_SIZE, 1)
        .with_input_dropout(0.1, true) // Variational dropout
        .with_recurrent_dropout(0.2, true) // Variational recurrent dropout
        .with_output_dropout(0.1); // Standard output dropout

    // Create warmup scheduler with cyclical base scheduler
    let base_scheduler =
        CyclicalLR::new(0.001, 0.01, DEMO_CYCLICAL_STEP_SIZE).with_mode(CyclicalMode::Triangular2);
    let warmup_scheduler = WarmupScheduler::new(DEMO_WARMUP_EPOCH_COUNT, base_scheduler, 0.0001);

    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::new(Adam::new(0.01), warmup_scheduler, 0.01);

    let config = advanced_training_config();

    let mut trainer =
        ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer).with_config(config);

    trainer.train(train_data, Some(val_data));

    println!("Final LR: {:.2e}", trainer.get_current_lr());
    println!(
        "Final Training Loss: {:.6}",
        trainer.get_latest_metrics().unwrap().train_loss
    );
    println!(
        "Final Validation Loss: {:.6}",
        trainer
            .get_latest_metrics()
            .unwrap()
            .validation_loss
            .unwrap()
    );

    println!("\n✅ Advanced training complete!");
}

pub fn generate_sine_wave_data(
    num_sequences: usize,
    offset: f64,
) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut data = Vec::new();

    for i in 0..num_sequences {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for t in 0..DEMO_SEQUENCE_LENGTH {
            let x = (offset + i as f64 * 0.1 + t as f64 * 0.2).sin();
            let y = (offset + i as f64 * 0.1 + (t + 1) as f64 * 0.2).sin();

            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }

        data.push((inputs, targets));
    }

    data
}

pub fn polynomial_decay_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_POLYNOMIAL_EPOCHS,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: true,
        early_stopping: None,
    }
}

pub fn cyclical_lr_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_CYCLICAL_EPOCHS,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None,
    }
}

pub fn warmup_scheduler_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_WARMUP_EPOCHS,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: true,
        early_stopping: None,
    }
}

pub fn advanced_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: DEMO_ADVANCED_EPOCHS,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_schedulers() {
        // Test polynomial scheduler
        let poly_scheduler = PolynomialLR::new(100, 2.0, 0.01);
        let schedule = LRScheduleVisualizer::generate_schedule(poly_scheduler, 0.1, 101);
        assert_eq!(schedule.len(), 101);
        assert_eq!(schedule[0].1, 0.1);
        assert!(schedule[99].1 < schedule[0].1);
        assert!((schedule[100].1 - 0.01).abs() < 1e-10);

        // Test cyclical scheduler
        let cyclical_scheduler = CyclicalLR::new(0.01, 0.1, 10);
        let schedule = LRScheduleVisualizer::generate_schedule(cyclical_scheduler, 0.01, 50);
        assert_eq!(schedule.len(), 50);
        assert_eq!(schedule[0].1, 0.01);

        // Test warmup scheduler
        let base_scheduler = rust_lstm::ConstantLR;
        let warmup_scheduler = WarmupScheduler::new(10, base_scheduler, 0.001);
        let schedule = LRScheduleVisualizer::generate_schedule(warmup_scheduler, 0.01, 20);
        assert_eq!(schedule.len(), 20);
        assert_eq!(schedule[0].1, 0.001);
        assert_eq!(schedule[10].1, 0.01);
    }
}
