use ndarray::arr2;
use rust_lstm::layers::linear::LinearLayer;
use rust_lstm::optimizers::{SGD, Adam};
use rust_lstm::models::lstm_network::LSTMNetwork;

/// Example 1: Basic LinearLayer usage for classification
fn basic_classification_example() {
    println!("=== Basic Classification Example ===");
    
    // Create a linear layer: 4 input features -> 3 classes
    let mut linear = LinearLayer::new(4, 3);
    let mut optimizer = SGD::new(0.1);
    
    // Sample input: batch of 2 samples, each with 4 features
    let input = arr2(&[
        [1.0, 0.5],  // feature 1
        [0.8, -0.2], // feature 2  
        [1.2, 0.9],  // feature 3
        [-0.1, 0.3]  // feature 4
    ]); // Shape: (4, 2)
    
    // Target classes (one-hot encoded)
    let targets = arr2(&[
        [1.0, 0.0],  // class 1 for sample 1, class 2 for sample 2
        [0.0, 1.0],  // 
        [0.0, 0.0]   //
    ]); // Shape: (3, 2)
    
    println!("Input shape: {:?}", input.shape());
    println!("Target shape: {:?}", targets.shape());
    
    // Training loop
    for epoch in 0..10 {
        // Forward pass
        let output = linear.forward(&input);
        
        // Simple loss: mean squared error
        let loss = (&output - &targets).map(|x| x * x).sum() / (output.len() as f64);
        
        // Backward pass
        let grad_output = 2.0 * (&output - &targets) / (output.len() as f64);
        let (gradients, _input_grad) = linear.backward(&grad_output);
        
        // Update parameters
        linear.update_parameters(&gradients, &mut optimizer, "classifier");
        
        if epoch % 2 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }
    
    // Final prediction
    let final_output = linear.forward(&input);
    println!("Final output:\n{:.3}", final_output);
    println!("Target:\n{:.3}", targets);
    println!();
}

/// Example 2: LSTM + LinearLayer for sequence classification
fn lstm_with_linear_example() {
    println!("=== LSTM + LinearLayer Example ===");
    
    // Create LSTM network: 5 input features -> 8 hidden units -> 3 classes
    let mut lstm = LSTMNetwork::new(5, 8, 1);
    let mut classifier = LinearLayer::new(8, 3);
    let mut optimizer = Adam::new(0.001);
    
    // Sample sequence data: 4 time steps, 5 features, batch size 1
    let sequence = vec![
        arr2(&[[1.0], [0.5], [0.2], [0.8], [0.1]]), // t=0
        arr2(&[[0.9], [0.6], [0.3], [0.7], [0.2]]), // t=1
        arr2(&[[0.8], [0.7], [0.4], [0.6], [0.3]]), // t=2
        arr2(&[[0.7], [0.8], [0.5], [0.5], [0.4]]), // t=3
    ];
    
    // Target: classify the entire sequence (shape: 3 classes, 1 sample)
    let target = arr2(&[[0.0], [1.0], [0.0]]); // Class 2
    
    println!("Sequence length: {}", sequence.len());
    println!("Input features: {}", sequence[0].nrows());
    println!("LSTM hidden size: {}", 8);
    println!("Output classes: {}", target.nrows());
    
    // Training loop
    for epoch in 0..20 {
        // LSTM forward pass
        let (lstm_outputs, _) = lstm.forward_sequence_with_cache(&sequence);
        
        // Use the last LSTM output for classification
        let last_hidden = &lstm_outputs.last().unwrap().0;
        
        // Linear layer forward pass
        let class_logits = classifier.forward(last_hidden);
        
        // Loss calculation
        let loss = (&class_logits - &target).map(|x| x * x).sum() / (class_logits.len() as f64);
        
        // Backward pass through linear layer
        let grad_output = 2.0 * (&class_logits - &target) / (class_logits.len() as f64);
        let (linear_grads, _lstm_grad) = classifier.backward(&grad_output);
        
        // Update linear layer
        classifier.update_parameters(&linear_grads, &mut optimizer, "classifier");
        
        // Note: In a complete implementation, you would also backpropagate through LSTM
        // This example focuses on demonstrating LinearLayer usage
        
        if epoch % 5 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }
    
    // Final prediction
    let (final_lstm_outputs, _) = lstm.forward_sequence_with_cache(&sequence);
    let final_hidden = &final_lstm_outputs.last().unwrap().0;
    let final_prediction = classifier.forward(final_hidden);
    
    println!("Final prediction: [{:.3}, {:.3}, {:.3}]", 
             final_prediction[[0, 0]], final_prediction[[1, 0]], final_prediction[[2, 0]]);
    println!("Target:           [{:.3}, {:.3}, {:.3}]", 
             target[[0, 0]], target[[1, 0]], target[[2, 0]]);
    println!();
}

/// Example 3: Multi-layer perceptron using multiple LinearLayers
fn multilayer_perceptron_example() {
    println!("=== Multi-Layer Perceptron Example ===");
    
    // Create a 3-layer MLP: 2 -> 4 -> 4 -> 1
    let mut layer1 = LinearLayer::new(2, 4);
    let mut layer2 = LinearLayer::new(4, 4);
    let mut layer3 = LinearLayer::new(4, 1);
    let mut optimizer = Adam::new(0.01);
    
    // XOR problem dataset
    let inputs = arr2(&[
        [0.0, 1.0, 0.0, 1.0], // input 1
        [0.0, 0.0, 1.0, 1.0]  // input 2
    ]); // Shape: (2, 4)
    
    let targets = arr2(&[[0.0, 1.0, 1.0, 0.0]]); // XOR outputs
    
    println!("Training MLP on XOR problem...");
    println!("Input shape: {:?}", inputs.shape());
    println!("Target shape: {:?}", targets.shape());
    
    // Training loop
    for epoch in 0..100 {
        // Forward pass
        let h1 = layer1.forward(&inputs);
        let h1_relu = h1.map(|&x| if x > 0.0 { x } else { 0.0 }); // ReLU activation
        
        let h2 = layer2.forward(&h1_relu);
        let h2_relu = h2.map(|&x| if x > 0.0 { x } else { 0.0 }); // ReLU activation
        
        let output = layer3.forward(&h2_relu);
        
        // Loss calculation
        let loss = (&output - &targets).map(|x| x * x).sum() / (output.len() as f64);
        
        // Backward pass
        let grad_output = 2.0 * (&output - &targets) / (output.len() as f64);
        
        // Layer 3 backward
        let (grad3, grad_h2) = layer3.backward(&grad_output);
        
        // ReLU backward for h2
        let grad_h2_relu = &grad_h2 * &h2.map(|&x| if x > 0.0 { 1.0 } else { 0.0 });
        
        // Layer 2 backward
        let (grad2, grad_h1) = layer2.backward(&grad_h2_relu);
        
        // ReLU backward for h1
        let grad_h1_relu = &grad_h1 * &h1.map(|&x| if x > 0.0 { 1.0 } else { 0.0 });
        
        // Layer 1 backward
        let (grad1, _) = layer1.backward(&grad_h1_relu);
        
        // Update all layers
        layer1.update_parameters(&grad1, &mut optimizer, "layer1");
        layer2.update_parameters(&grad2, &mut optimizer, "layer2");
        layer3.update_parameters(&grad3, &mut optimizer, "layer3");
        
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }
    
    // Final predictions
    let h1 = layer1.forward(&inputs);
    let h1_relu = h1.map(|&x| if x > 0.0 { x } else { 0.0 });
    let h2 = layer2.forward(&h1_relu);
    let h2_relu = h2.map(|&x| if x > 0.0 { x } else { 0.0 });
    let final_output = layer3.forward(&h2_relu);
    
    println!("Final predictions:");
    for i in 0..4 {
        let input_vals = (inputs[[0, i]], inputs[[1, i]]);
        let prediction = final_output[[0, i]];
        let target_val = targets[[0, i]];
        println!("  {:?} -> {:.3} (target: {:.1})", input_vals, prediction, target_val);
    }
    println!();
}

/// Example 4: Demonstrating different initialization methods
fn initialization_example() {
    println!("=== Initialization Methods Example ===");
    
    // Method 1: Default random initialization (Xavier/Glorot)
    let layer_random = LinearLayer::new(3, 2);
    println!("Random initialization:");
    println!("  Weight range: [{:.3}, {:.3}]", 
             layer_random.weight.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
             layer_random.weight.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
    
    // Method 2: Zero initialization
    let layer_zeros = LinearLayer::new_zeros(3, 2);
    println!("Zero initialization:");
    println!("  All weights: {}", layer_zeros.weight.iter().all(|&x| x == 0.0));
    
    // Method 3: Custom initialization
    let custom_weights = arr2(&[[1.0, 0.5, -0.2], [0.8, -0.1, 0.3]]);
    let custom_bias = arr2(&[[0.1], [-0.05]]);
    let layer_custom = LinearLayer::from_weights(custom_weights.clone(), custom_bias.clone());
    println!("Custom initialization:");
    println!("  Custom weights shape: {:?}", layer_custom.weight.shape());
    println!("  Custom bias shape: {:?}", layer_custom.bias.shape());
    
    // Show layer information
    println!("Layer dimensions: {:?}", layer_custom.dimensions());
    println!("Number of parameters: {}", layer_custom.num_parameters());
    println!();
}

fn main() {
    println!("LinearLayer Examples");
    println!("===================\n");
    
    basic_classification_example();
    lstm_with_linear_example();
    multilayer_perceptron_example();
    initialization_example();
    
    println!("All examples completed successfully! 🎉");
    println!("\nKey takeaways:");
    println!("- LinearLayer enables standard neural network architectures");
    println!("- Works seamlessly with LSTM networks for classification");
    println!("- Supports multiple initialization methods");
    println!("- Integrates with all existing optimizers");
    println!("- Essential for text generation and classification tasks");
}
