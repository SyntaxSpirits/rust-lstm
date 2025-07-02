use rust_lstm::{GRUNetwork, GRULayerDropoutConfig, Adam, MSELoss, LossFunction};
use ndarray::{Array2, arr2, s};

fn main() {
    println!("🧠 GRU Network Example");
    println!("=====================");

    // Example 1: Basic GRU forward pass
    basic_gru_example();

    // Example 2: Multi-layer GRU with dropout
    multilayer_gru_example();

    // Example 3: GRU sequence modeling
    sequence_modeling_example();

    // Example 4: Simple training example
    simple_training_example();
}

fn basic_gru_example() {
    println!("\n📝 1. Basic GRU Forward Pass");
    
    let input_size = 3;
    let hidden_size = 4;
    let mut gru = GRUNetwork::new(input_size, hidden_size, 1);
    
    let input = arr2(&[[1.0], [0.5], [-0.3]]);
    let hidden_state = vec![Array2::zeros((hidden_size, 1))];
    
    let output = gru.forward(&input, &hidden_state);
    
    println!("   Input shape: {:?}", input.shape());
    println!("   Output shape: {:?}", output[0].shape());
    println!("   First few output values: {:?}", output[0].slice(s![0..2, 0]));
}

fn multilayer_gru_example() {
    println!("\n📝 2. Multi-layer GRU with Dropout");
    
    let input_size = 5;
    let hidden_size = 8;
    let num_layers = 3;
    
    // Create GRU with different dropout configurations per layer
    let layer_configs = vec![
        GRULayerDropoutConfig::new()
            .with_input_dropout(0.2, true)
            .with_recurrent_dropout(0.3, false),
        GRULayerDropoutConfig::new()
            .with_input_dropout(0.1, false)
            .with_recurrent_dropout(0.2, true)
            .with_output_dropout(0.1),
        GRULayerDropoutConfig::new()
            .with_recurrent_dropout(0.15, false),
    ];
    
    let mut gru = GRUNetwork::new(input_size, hidden_size, num_layers)
        .with_layer_dropout(layer_configs);
    
    let input = Array2::from_shape_fn((input_size, 1), |_| rand::random::<f64>() * 2.0 - 1.0);
    let hidden_states: Vec<Array2<f64>> = (0..num_layers)
        .map(|_| Array2::zeros((hidden_size, 1)))
        .collect();
    
    // Test in training mode
    gru.train();
    let outputs_train = gru.forward(&input, &hidden_states);
    
    // Test in evaluation mode
    gru.eval();
    let outputs_eval = gru.forward(&input, &hidden_states);
    
    println!("   Network: {} layers, {} input size, {} hidden size", num_layers, input_size, hidden_size);
    println!("   Training mode outputs shape: {:?}", outputs_train.last().unwrap().shape());
    println!("   Evaluation mode outputs shape: {:?}", outputs_eval.last().unwrap().shape());
    println!("   All layer outputs count: {}", outputs_train.len());
}

fn sequence_modeling_example() {
    println!("\n📝 3. GRU Sequence Modeling");
    
    let input_size = 2;
    let hidden_size = 6;
    let sequence_length = 5;
    
    let mut gru = GRUNetwork::new(input_size, hidden_size, 2)
        .with_input_dropout(0.1, true)
        .with_recurrent_dropout(0.2, false);
    
    // Create a simple sequence (sine wave pattern)
    let mut sequence = Vec::new();
    for i in 0..sequence_length {
        let t = i as f64 * 0.1;
        let input = arr2(&[[t.sin()], [t.cos()]]);
        sequence.push(input);
    }
    
    gru.train();
    let (outputs, _caches) = gru.forward_sequence_with_cache(&sequence);
    
    println!("   Sequence length: {}", sequence_length);
    println!("   Input size: {}, Hidden size: {}", input_size, hidden_size);
    println!("   Output sequence length: {}", outputs.len());
    
    for (i, (output, _layer_outputs)) in outputs.iter().enumerate() {
        println!("   Step {} output norm: {:.4}", i, output.iter().map(|&x| x * x).sum::<f64>().sqrt());
    }
}

fn simple_training_example() {
    println!("\n📝 4. Simple Training Example");
    
    let input_size = 2;
    let hidden_size = 4;
    let mut gru = GRUNetwork::new(input_size, hidden_size, 1);
    
    let mut optimizer = Adam::new(0.001);
    let loss_fn = MSELoss;
    
    // Simple training data: predict next value in sequence
    let train_sequences = vec![
        (
            vec![arr2(&[[0.0], [1.0]]), arr2(&[[0.5], [0.5]])],
            vec![arr2(&[[1.0]]), arr2(&[[0.0]])]
        ),
        (
            vec![arr2(&[[1.0], [0.0]]), arr2(&[[-0.5], [0.5]])],
            vec![arr2(&[[0.0]]), arr2(&[[1.0]])]
        ),
    ];
    
    println!("   Training for 5 epochs...");
    
    for epoch in 0..5 {
        let mut total_loss = 0.0;
        
        for (inputs, targets) in &train_sequences {
            // Forward pass
            let (outputs, caches) = gru.forward_sequence_with_cache(inputs);
            
            // Compute loss
            let mut sequence_loss = 0.0;
            let mut gradients_accum = gru.zero_gradients();
            
            for (step, ((output, _), target)) in outputs.iter().zip(targets.iter()).enumerate() {
                let step_loss = loss_fn.compute_loss(output, target);
                sequence_loss += step_loss;
                
                let dloss = loss_fn.compute_gradient(output, target);
                let (step_gradients, _) = gru.backward(&dloss, &caches[step]);
                
                // Accumulate gradients
                for (acc_grad, step_grad) in gradients_accum.iter_mut().zip(step_gradients.iter()) {
                    acc_grad.w_ir += &step_grad.w_ir;
                    acc_grad.w_hr += &step_grad.w_hr;
                    acc_grad.b_ir += &step_grad.b_ir;
                    acc_grad.b_hr += &step_grad.b_hr;
                    acc_grad.w_iz += &step_grad.w_iz;
                    acc_grad.w_hz += &step_grad.w_hz;
                    acc_grad.b_iz += &step_grad.b_iz;
                    acc_grad.b_hz += &step_grad.b_hz;
                    acc_grad.w_ih += &step_grad.w_ih;
                    acc_grad.w_hh += &step_grad.w_hh;
                    acc_grad.b_ih += &step_grad.b_ih;
                    acc_grad.b_hh += &step_grad.b_hh;
                }
            }
            
            // Update parameters
            gru.update_parameters(&gradients_accum, &mut optimizer);
            total_loss += sequence_loss;
        }
        
        println!("   Epoch {}: Average Loss = {:.6}", epoch + 1, total_loss / train_sequences.len() as f64);
    }
    
    println!("   Training completed!");
} 