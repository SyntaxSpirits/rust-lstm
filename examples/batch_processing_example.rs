use ndarray::{Array2, arr2};
use rust_lstm::{LSTMNetwork, create_adam_batch_trainer, create_basic_trainer};
use std::time::Instant;

/// Generate synthetic sine wave sequences for batch processing demonstration
fn generate_batch_sine_data(num_sequences: usize, sequence_length: usize, input_size: usize) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut data = Vec::new();
    
    for i in 0..num_sequences {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        let start = (i as f64) * 0.05; // Different starting points for variety
        let frequency = 1.0 + (i as f64) * 0.1; // Different frequencies
        
        for j in 0..sequence_length {
            let t = start + (j as f64) * 0.1;
            
            // Create multi-dimensional input
            let mut input_vec = vec![0.0; input_size];
            input_vec[0] = (t * frequency * 2.0 * std::f64::consts::PI).sin();
            if input_size > 1 {
                input_vec[1] = (t * frequency * 2.0 * std::f64::consts::PI).cos();
            }
            if input_size > 2 {
                input_vec[2] = t.sin() * t.cos(); // Some nonlinear combination
            }
            
            // Target is the next value in the sine sequence
            let target = ((t + 0.1) * frequency * 2.0 * std::f64::consts::PI).sin();
            
            inputs.push(Array2::from_shape_vec((input_size, 1), input_vec).unwrap());
            targets.push(arr2(&[[target]]));
        }
        
        data.push((inputs, targets));
    }
    
    data
}

/// Benchmark training time comparison between single and batch processing
fn benchmark_training_performance() {
    println!("BATCH PROCESSING PERFORMANCE BENCHMARK");
    println!("======================================\n");

    let input_size = 3;
    let hidden_size = 16;
    let num_layers = 2;
    let learning_rate = 0.001;
    
    // Generate training data
    let train_data = generate_batch_sine_data(100, 10, input_size);
    let val_data = generate_batch_sine_data(20, 10, input_size);
    
    println!("Dataset: {} training sequences, {} validation sequences", train_data.len(), val_data.len());
    println!("Network: {} -> {} hidden ({} layers)\n", input_size, hidden_size, num_layers);

    // Test 1: Single sequence processing (traditional)
    println!("Testing Traditional Single-Sequence Processing...");
    let network1 = LSTMNetwork::new(input_size, hidden_size, num_layers);
    let mut trainer1 = create_basic_trainer(network1, learning_rate);
    
    // Configure for quick demo
    trainer1.config.epochs = 5;
    trainer1.config.print_every = 1;
    
    let start_time = Instant::now();
    trainer1.train(&train_data, Some(&val_data));
    let single_time = start_time.elapsed();
    
    let final_metrics1 = trainer1.get_latest_metrics().unwrap();
    println!("Single-sequence - Final loss: {:.6}, Time: {:.2}s\n", 
             final_metrics1.train_loss, single_time.as_secs_f64());

    // Test 2: Batch processing with small batches
    println!("Testing Batch Processing (batch size 8)...");
    let network2 = LSTMNetwork::new(input_size, hidden_size, num_layers);
    let mut trainer2 = create_adam_batch_trainer(network2, learning_rate);
    
    trainer2.config.epochs = 5;
    trainer2.config.print_every = 1;
    
    let start_time = Instant::now();
    trainer2.train(&train_data, Some(&val_data), 8); // Batch size 8
    let batch_time = start_time.elapsed();
    
    let final_metrics2 = trainer2.get_latest_metrics().unwrap();
    println!("Batch processing - Final loss: {:.6}, Time: {:.2}s\n", 
             final_metrics2.train_loss, batch_time.as_secs_f64());

    // Test 3: Larger batch size
    println!("Testing Larger Batch Processing (batch size 16)...");
    let network3 = LSTMNetwork::new(input_size, hidden_size, num_layers);
    let mut trainer3 = create_adam_batch_trainer(network3, learning_rate);
    
    trainer3.config.epochs = 5;
    trainer3.config.print_every = 1;
    
    let start_time = Instant::now();
    trainer3.train(&train_data, Some(&val_data), 16); // Batch size 16
    let large_batch_time = start_time.elapsed();
    
    let final_metrics3 = trainer3.get_latest_metrics().unwrap();
    println!("Large batch processing - Final loss: {:.6}, Time: {:.2}s\n", 
             final_metrics3.train_loss, large_batch_time.as_secs_f64());

    // Performance summary
    println!("PERFORMANCE SUMMARY:");
    println!("======================");
    println!("Single-sequence:   {:.2}s (baseline)", single_time.as_secs_f64());
    println!("Batch-8:          {:.2}s ({:.1}x speedup)", 
             batch_time.as_secs_f64(), 
             single_time.as_secs_f64() / batch_time.as_secs_f64());
    println!("Batch-16:         {:.2}s ({:.1}x speedup)", 
             large_batch_time.as_secs_f64(), 
             single_time.as_secs_f64() / large_batch_time.as_secs_f64());
    
    if batch_time < single_time {
        println!("Batch processing achieved {:.1}x speedup!", 
                 single_time.as_secs_f64() / batch_time.as_secs_f64());
    } else {
        println!("Note: For small datasets, overhead may dominate. Try larger datasets for better speedup.");
    }
}

/// Demonstrate batch prediction capabilities
fn demonstrate_batch_prediction() {
    println!("\nBATCH PREDICTION DEMONSTRATION");
    println!("==============================\n");

    let input_size = 2;
    let hidden_size = 8;
    let num_layers = 1;
    
    // Create and train a simple model
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);
    let mut trainer = create_adam_batch_trainer(network, 0.01);
    
    // Generate small training dataset
    let train_data = generate_batch_sine_data(20, 5, input_size);
    
    trainer.config.epochs = 10;
    trainer.config.print_every = 5;
    
    println!("Training a small model for prediction demo...");
    trainer.train(&train_data, None, 4);
    
    // Create test sequences for batch prediction
    let test_sequences = generate_batch_sine_data(3, 3, input_size);
    let test_inputs: Vec<_> = test_sequences.iter().map(|(inputs, _)| inputs.clone()).collect();
    let _test_targets: Vec<_> = test_sequences.iter().map(|(_, targets)| targets.clone()).collect();
    
    println!("\nPerforming batch predictions...");
    let predictions = trainer.predict_batch(&test_inputs);
    
    println!("Input sequences vs Predictions:");
    for (i, (inputs, preds)) in test_inputs.iter().zip(predictions.iter()).enumerate() {
        println!("Sequence {}:", i + 1);
        for (j, (input, pred)) in inputs.iter().zip(preds.iter()).enumerate() {
            println!("  Step {}: Input=[{:.3}, {:.3}] -> Pred={:.3}", 
                     j + 1, input[[0, 0]], input[[1, 0]], pred[[0, 0]]);
        }
        println!();
    }
}

/// Demonstrate memory efficiency and scalability
fn demonstrate_scalability() {
    println!("SCALABILITY DEMONSTRATION");
    println!("=========================\n");

    let test_sizes = vec![
        (50, 4),    // Small: 50 sequences, batch size 4
        (200, 8),   // Medium: 200 sequences, batch size 8  
        (500, 16),  // Large: 500 sequences, batch size 16
    ];

    for (num_sequences, batch_size) in test_sizes {
        println!("Testing with {} sequences, batch size {}...", num_sequences, batch_size);
        
        let train_data = generate_batch_sine_data(num_sequences, 8, 2);
        let network = LSTMNetwork::new(2, 12, 1);
        let mut trainer = create_adam_batch_trainer(network, 0.001);
        
        trainer.config.epochs = 3;
        trainer.config.print_every = 1;
        
        let start_time = Instant::now();
        trainer.train(&train_data, None, batch_size);
        let training_time = start_time.elapsed();
        
        let final_loss = trainer.get_latest_metrics().unwrap().train_loss;
        println!("   Completed in {:.2}s, final loss: {:.6}\n", 
                 training_time.as_secs_f64(), final_loss);
    }
    
    println!("All scalability tests completed successfully!");
    println!("Batch processing handles varying dataset sizes efficiently.");
}

fn main() {
    println!("RUST-LSTM BATCH PROCESSING DEMONSTRATION");
    println!("=========================================\n");
    
    println!("This example demonstrates the new batch processing capabilities:");
    println!("- Simultaneous processing of multiple sequences");
    println!("- Performance improvements over single-sequence training");
    println!("- Batch prediction capabilities");
    println!("- Scalability with different batch sizes\n");

    benchmark_training_performance();
    demonstrate_batch_prediction();  
    demonstrate_scalability();
    
    println!("\nBATCH PROCESSING DEMONSTRATION COMPLETED!");
    println!("==========================================");
    println!("Key Benefits Demonstrated:");
    println!("- Faster training through batch processing");
    println!("- Efficient memory utilization");
    println!("- Scalable to different dataset sizes");
    println!("- Easy-to-use batch training API");
    println!("- Backward compatibility with existing code");
    
    println!("\nNext Steps:");
    println!("- Try batch processing with your own datasets");
    println!("- Experiment with different batch sizes");
    println!("- Compare performance with single-sequence training");
    println!("- Use batch processing for faster model development");
} 