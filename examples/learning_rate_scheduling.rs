use ndarray::{Array2, arr2};
use rust_lstm::{
    LSTMNetwork, ScheduledLSTMTrainer, ScheduledOptimizer, TrainingConfig,
    Adam, MSELoss, optimizers::Optimizer,
    ReduceLROnPlateau,
    create_step_lr_trainer, create_one_cycle_trainer, create_cosine_annealing_trainer
};

fn main() {
    println!("Learning Rate Scheduling Examples for Rust-LSTM");
    println!("==================================================\n");

    // Generate sample training data (sine wave prediction)
    let train_data = generate_sine_wave_data(100, 0.0);
    let val_data = generate_sine_wave_data(20, 1000.0);
    
    // Example 1: Step Learning Rate Decay
    step_lr_example(&train_data, &val_data);
    
    // Example 2: OneCycle Learning Rate Policy
    one_cycle_example(&train_data, &val_data);
    
    // Example 3: Cosine Annealing
    cosine_annealing_example(&train_data, &val_data);
    
    // Example 4: Exponential Decay
    exponential_decay_example(&train_data, &val_data);
    
    // Example 5: ReduceLROnPlateau (manual stepping)
    reduce_on_plateau_example(&train_data, &val_data);
    
    // Example 6: Comparison of different schedulers
    scheduler_comparison(&train_data, &val_data);
}

fn step_lr_example(train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                   val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) {
    println!("Step Learning Rate Decay Example");
    println!("Reduces LR by factor of 0.5 every 10 epochs\n");
    
    let network = LSTMNetwork::new(1, 10, 2)
        .with_input_dropout(0.1, false)
        .with_recurrent_dropout(0.2, true);
    
    let config = TrainingConfig {
        epochs: 30,
        print_every: 5,
        clip_gradient: Some(1.0),
        log_lr_changes: true,
        early_stopping: None,
    };
    
    let mut trainer = create_step_lr_trainer(network, 0.01, 10, 0.5)
        .with_config(config);
    
    trainer.train(train_data, Some(val_data));
    
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());
    println!("----------------------------------------\n");
}

fn one_cycle_example(train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                     val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) {
    println!("OneCycle Learning Rate Policy Example");
    println!("Starts low, ramps up to max, then anneals down\n");
    
    let network = LSTMNetwork::new(1, 10, 2);
    
    let config = TrainingConfig {
        epochs: 50,
        print_every: 10,
        clip_gradient: Some(1.0),
        log_lr_changes: false, // Too many changes for OneCycle
        early_stopping: None,
    };
    
    let mut trainer = create_one_cycle_trainer(network, 0.1, 50)
        .with_config(config);
    
    trainer.train(train_data, Some(val_data));
    
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());
    println!("----------------------------------------\n");
}

fn cosine_annealing_example(train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                           val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) {
    println!("Cosine Annealing Example");
    println!("Smoothly oscillates LR following cosine curve\n");
    
    let network = LSTMNetwork::new(1, 10, 2);
    
    let config = TrainingConfig {
        epochs: 40,
        print_every: 8,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None,
    };
    
    let mut trainer = create_cosine_annealing_trainer(network, 0.01, 20, 1e-6)
        .with_config(config);
    
    trainer.train(train_data, Some(val_data));
    
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());
    println!("----------------------------------------\n");
}

fn exponential_decay_example(train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                            val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) {
    println!("Exponential Decay Example");
    println!("Continuously decays LR by factor of 0.95 each epoch\n");
    
    let network = LSTMNetwork::new(1, 10, 2);
    
    let loss_function = MSELoss;
    let scheduled_optimizer = ScheduledOptimizer::exponential(
        Adam::new(0.01), 
        0.01, 
        0.95
    );
    
    let config = TrainingConfig {
        epochs: 30,
        print_every: 6,
        clip_gradient: Some(1.0),
        log_lr_changes: true,
        early_stopping: None,
    };
    
    let mut trainer = ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer)
        .with_config(config);
    
    trainer.train(train_data, Some(val_data));
    
    println!("Final LR: {:.2e}\n", trainer.get_current_lr());
    println!("----------------------------------------\n");
}

fn reduce_on_plateau_example(train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                            val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) {
    println!("ReduceLROnPlateau Example");
    println!("Reduces LR when validation loss stops improving\n");
    
    let network = LSTMNetwork::new(1, 10, 2);
    
    // Create a plateau scheduler manually since we need special handling
    let mut plateau_scheduler = ReduceLROnPlateau::new(0.5, 5);
    let mut optimizer = Adam::new(0.01);
    let loss_function = MSELoss;
    
    let config = TrainingConfig {
        epochs: 40,
        print_every: 5,
        clip_gradient: Some(1.0),
        log_lr_changes: true,
        early_stopping: None,
    };
    
    println!("Training with manual ReduceLROnPlateau stepping...");
    
    // Manual training loop for ReduceLROnPlateau
    for epoch in 0..config.epochs {
        // Simulate training loss (would be actual training in real scenario)
        let train_loss = 0.1 * (-(epoch as f64) * 0.05).exp();
        
        // Simulate validation loss with some noise
        let val_loss = train_loss + 0.01 * (epoch as f64 * 0.1).sin();
        
        // Step the plateau scheduler with validation loss
        let new_lr = plateau_scheduler.step(val_loss, 0.01);
        optimizer.set_learning_rate(new_lr);
        
        if epoch % config.print_every == 0 {
            println!("Epoch {}: Train Loss: {:.6}, Val Loss: {:.6}, LR: {:.2e}", 
                     epoch, train_loss, val_loss, new_lr);
        }
    }
    
    println!("\nFinal LR: {:.2e}\n", optimizer.get_learning_rate());
    println!("----------------------------------------\n");
}

fn scheduler_comparison(train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                       val_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) {
    println!("Scheduler Comparison");
    println!("Training the same network with different schedulers\n");
    
    let schedulers = vec![
        ("Constant", "constant"),
        ("StepLR", "step"), 
        ("Exponential", "exp"),
        ("OneCycle", "onecycle"),
    ];
    
    for (name, scheduler_type) in schedulers {
        println!("Testing {} scheduler:", name);
        
        let network = LSTMNetwork::new(1, 8, 1); // Smaller network for faster comparison
        
        let config = TrainingConfig {
            epochs: 20,
            print_every: 20, // Only print final result
            clip_gradient: Some(1.0),
            log_lr_changes: false,
            early_stopping: None,
        };
        
        let final_loss = match scheduler_type {
            "constant" => {
                let mut trainer = create_step_lr_trainer(network, 0.01, 1000, 1.0) // Effectively constant
                    .with_config(config);
                trainer.train(train_data, Some(val_data));
                trainer.get_latest_metrics().unwrap().validation_loss.unwrap_or(0.0)
            },
            "step" => {
                let mut trainer = create_step_lr_trainer(network, 0.01, 10, 0.5)
                    .with_config(config);
                trainer.train(train_data, Some(val_data));
                trainer.get_latest_metrics().unwrap().validation_loss.unwrap_or(0.0)
            },
            "exp" => {
                let loss_function = MSELoss;
                let scheduled_optimizer = ScheduledOptimizer::exponential(
                    Adam::new(0.01), 0.01, 0.95
                );
                let mut trainer = ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer)
                    .with_config(config);
                trainer.train(train_data, Some(val_data));
                trainer.get_latest_metrics().unwrap().validation_loss.unwrap_or(0.0)
            },
            "onecycle" => {
                let mut trainer = create_one_cycle_trainer(network, 0.05, 20)
                    .with_config(config);
                trainer.train(train_data, Some(val_data));
                trainer.get_latest_metrics().unwrap().validation_loss.unwrap_or(0.0)
            },
            _ => 0.0,
        };
        
        println!("   Final validation loss: {:.6}\n", final_loss);
    }
    
    println!("Comparison complete! Check which scheduler performed best.");
}

fn generate_sine_wave_data(num_sequences: usize, offset: f64) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut data = Vec::new();
    
    for i in 0..num_sequences {
        let sequence_length = 10;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        for t in 0..sequence_length {
            let x = (offset + i as f64 * 0.1 + t as f64 * 0.2).sin();
            let y = (offset + i as f64 * 0.1 + (t + 1) as f64 * 0.2).sin(); // Next value
            
            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }
        
        data.push((inputs, targets));
    }
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_lstm::{SGD, StepLR};

    #[test]
    fn test_scheduler_creation() {
        let network = LSTMNetwork::new(2, 4, 1);
        
        // Test step LR creation
        let trainer = create_step_lr_trainer(network.clone(), 0.01, 10, 0.5);
        assert_eq!(trainer.get_current_lr(), 0.01);
        
        // Test one cycle creation
        let trainer = create_one_cycle_trainer(network.clone(), 0.1, 100);
        assert!(trainer.get_current_lr() > 0.0);
        
        // Test cosine annealing creation
        let trainer = create_cosine_annealing_trainer(network, 0.01, 50, 1e-6);
        assert_eq!(trainer.get_current_lr(), 0.01);
    }

    #[test]
    fn test_manual_scheduler() {
        let network = LSTMNetwork::new(2, 4, 1);
        let loss_function = MSELoss;
        let scheduled_optimizer = ScheduledOptimizer::new(
            SGD::new(0.01),
            StepLR::new(5, 0.5),
            0.01
        );
        
        let trainer = ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer);
        assert_eq!(trainer.get_current_lr(), 0.01);
        assert_eq!(trainer.get_current_epoch(), 0);
    }
} 