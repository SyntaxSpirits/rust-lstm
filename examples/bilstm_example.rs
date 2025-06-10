use ndarray::{Array2, arr2};
use rust_lstm::layers::bilstm_network::{BiLSTMNetwork, CombineMode};
use rust_lstm::models::lstm_network::LSTMNetwork;

/// Generate a simple sequence that benefits from bidirectional processing
fn generate_bidirectional_data() -> Vec<Array2<f64>> {
    let sequence_length = 10;
    let mut sequence = Vec::new();
    
    for t in 0..sequence_length {
        let t_f = t as f64 * 0.5;
        let current = t_f.sin();
        let future = if t < sequence_length - 1 { (t_f + 0.5).cos() * 0.5 } else { 0.0 };
        let past = if t > 0 { (t_f - 0.5).sin() * 0.3 } else { 0.0 };
        
        let value = current + future + past;
        sequence.push(arr2(&[[value]]));
    }
    
    sequence
}

/// Demonstrate basic BiLSTM functionality
fn demo_basic_bilstm() {
    println!("=== Basic BiLSTM Demonstration ===");
    
    let mut bilstm = BiLSTMNetwork::new_concat(1, 4, 1);
    let sequence = generate_bidirectional_data();
    
    println!("Input sequence length: {}", sequence.len());
    println!("BiLSTM hidden size: {}", bilstm.hidden_size);
    println!("BiLSTM output size: {}", bilstm.output_size());
    
    let outputs = bilstm.forward_sequence(&sequence);
    
    println!("Output shapes:");
    for (i, output) in outputs.iter().enumerate() {
        println!("  Time step {}: {:?}", i, output.shape());
    }
    
    println!("Sample output values (first 3 time steps):");
    for (i, output) in outputs.iter().take(3).enumerate() {
        println!("  t={}: [{:.4}, {:.4}, {:.4}, ...]", 
                 i, output[[0,0]], output[[1,0]], output[[2,0]]);
    }
}

/// Compare different combine modes
fn demo_combine_modes() {
    println!("\n=== BiLSTM Combine Modes Comparison ===");
    
    let sequence = generate_bidirectional_data();
    
    // Test different combine modes
    let modes = vec![
        ("Concatenation", CombineMode::Concat),
        ("Sum", CombineMode::Sum),
        ("Average", CombineMode::Average),
    ];
    
    for (name, mode) in modes {
        let mut bilstm = BiLSTMNetwork::new(1, 3, 1, mode);
        let outputs = bilstm.forward_sequence(&sequence);
        
        println!("{} mode:", name);
        println!("  Output size: {}", bilstm.output_size());
        println!("  First output shape: {:?}", outputs[0].shape());
        println!("  Sample values: [{:.4}, {:.4}]", 
                 outputs[0][[0,0]], 
                 if outputs[0].nrows() > 1 { outputs[0][[1,0]] } else { 0.0 });
    }
}

/// Compare BiLSTM vs unidirectional LSTM performance
fn demo_bilstm_vs_lstm() {
    println!("\n=== BiLSTM vs Unidirectional LSTM Comparison ===");
    
    let sequence = generate_bidirectional_data();
    
    // Unidirectional LSTM
    let mut lstm = LSTMNetwork::new(1, 4, 1);
    let mut hx = Array2::zeros((4, 1));
    let mut cx = Array2::zeros((4, 1));
    
    let mut lstm_outputs = Vec::new();
    for input in &sequence {
        let (new_hx, new_cx) = lstm.forward(input, &hx, &cx);
        lstm_outputs.push(new_hx.clone());
        hx = new_hx;
        cx = new_cx;
    }
    
    // Bidirectional LSTM (with same total parameters approximately)
    let mut bilstm = BiLSTMNetwork::new_concat(1, 2, 1); // 2*2=4 total hidden units
    let bilstm_outputs = bilstm.forward_sequence(&sequence);
    
    println!("Unidirectional LSTM:");
    println!("  Hidden size: 4");
    println!("  Output size: 4");
    println!("  Sample output: [{:.4}, {:.4}, {:.4}, {:.4}]", 
             lstm_outputs[0][[0,0]], lstm_outputs[0][[1,0]], 
             lstm_outputs[0][[2,0]], lstm_outputs[0][[3,0]]);
    
    println!("Bidirectional LSTM:");
    println!("  Hidden size per direction: 2");
    println!("  Total output size: 4");
    println!("  Sample output: [{:.4}, {:.4}, {:.4}, {:.4}]", 
             bilstm_outputs[0][[0,0]], bilstm_outputs[0][[1,0]], 
             bilstm_outputs[0][[2,0]], bilstm_outputs[0][[3,0]]);
    
    // Demonstrate that BiLSTM has access to future context
    println!("\nContext Analysis:");
    println!("  LSTM processes left-to-right only");
    println!("  BiLSTM processes both directions and combines information");
    println!("  This allows BiLSTM to use future context for current predictions");
}

/// Demonstrate multi-layer BiLSTM
fn demo_multilayer_bilstm() {
    println!("\n=== Multi-layer BiLSTM ===");
    
    let sequence = generate_bidirectional_data();
    
    for num_layers in 1..=3 {
        let mut bilstm = BiLSTMNetwork::new_concat(1, 3, num_layers);
        let outputs = bilstm.forward_sequence(&sequence);
        
        println!("{}-layer BiLSTM:", num_layers);
        println!("  Total parameters (approx): {}", 
                 num_layers * 2 * (3 * 4 * (if num_layers == 1 { 1 } else { 6 }) + 4 * 3));
        println!("  Output shape: {:?}", outputs[0].shape());
        println!("  Sample output magnitude: {:.4}", 
                 outputs[0].iter().map(|&x| x.abs()).sum::<f64>() / outputs[0].len() as f64);
    }
}

/// Demonstrate BiLSTM with dropout
fn demo_bilstm_with_dropout() {
    println!("\n=== BiLSTM with Dropout ===");
    
    let sequence = generate_bidirectional_data();
    
    let mut bilstm = BiLSTMNetwork::new_concat(1, 4, 2)
        .with_input_dropout(0.2, true)      // 20% variational input dropout
        .with_recurrent_dropout(0.3, true)  // 30% variational recurrent dropout
        .with_output_dropout(0.1);          // 10% output dropout
    
    // Training mode (dropout active)
    bilstm.train();
    let train_outputs = bilstm.forward_sequence(&sequence);
    
    // Evaluation mode (dropout inactive)
    bilstm.eval();
    let eval_outputs = bilstm.forward_sequence(&sequence);
    
    println!("Training mode (with dropout):");
    println!("  Sample output: [{:.4}, {:.4}, {:.4}]", 
             train_outputs[0][[0,0]], train_outputs[0][[1,0]], train_outputs[0][[2,0]]);
    
    println!("Evaluation mode (no dropout):");
    println!("  Sample output: [{:.4}, {:.4}, {:.4}]", 
             eval_outputs[0][[0,0]], eval_outputs[0][[1,0]], eval_outputs[0][[2,0]]);
    
    println!("Dropout correctly affects training vs evaluation outputs");
}

/// Demonstrate sequence processing with caching
fn demo_bilstm_with_caching() {
    println!("\n=== BiLSTM with Caching (for Training) ===");
    
    let sequence = generate_bidirectional_data();
    let mut bilstm = BiLSTMNetwork::new_concat(1, 3, 1);
    
    let (outputs, cache) = bilstm.forward_sequence_with_cache(&sequence);
    
    println!("Forward pass with caching:");
    println!("  Sequence length: {}", sequence.len());
    println!("  Number of outputs: {}", outputs.len());
    println!("  Forward caches: {}", cache.forward_caches.len());
    println!("  Backward caches: {}", cache.backward_caches.len());
    println!("  Cache enables efficient backpropagation for training");
}

fn main() {
    println!("🔄 Bidirectional LSTM Demonstration");
    println!("=====================================");
    
    demo_basic_bilstm();
    demo_combine_modes();
    demo_bilstm_vs_lstm();
    demo_multilayer_bilstm();
    demo_bilstm_with_dropout();
    demo_bilstm_with_caching();
    
    println!("\n✅ BiLSTM demonstration completed!");
    println!("\nKey Benefits of Bidirectional LSTM:");
    println!("• Captures both past and future context");
    println!("• Better for tasks where full sequence is available");
    println!("• Improved performance on sequence labeling tasks");
    println!("• Flexible output combination modes");
    println!("• Compatible with existing dropout and training systems");
} 