# Rust-LSTM

A comprehensive LSTM (Long Short-Term Memory) neural network library implemented in Rust with complete training capabilities, multiple optimizers, and advanced regularization.

## Features

- **LSTM, BiLSTM & GRU Networks** with multi-layer support
- **Complete Training System** with backpropagation through time (BPTT)
- **Multiple Optimizers**: SGD, Adam, RMSprop with learning rate scheduling
- **Loss Functions**: MSE, MAE, Cross-entropy with softmax
- **Advanced Dropout**: Input, recurrent, output dropout, variational dropout, and zoneout
- **Model Persistence**: Save/load models in JSON or binary format
- **Peephole LSTM variant** for enhanced performance

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-lstm = "0.3.0"
```

### Basic Usage

```rust
use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    // Create LSTM network
    let mut network = LSTMNetwork::new(3, 10, 2); // input_size, hidden_size, num_layers
    
    // Create input data
    let input = Array2::from_shape_vec((3, 1), vec![0.5, 0.1, -0.3]).unwrap();
    let hx = Array2::zeros((10, 1));
    let cx = Array2::zeros((10, 1));
    
    // Forward pass
    let (output, _) = network.forward(&input, &hx, &cx);
    println!("Output: {:?}", output);
}
```

### Training Example

```rust
use rust_lstm::{LSTMNetwork, create_basic_trainer, TrainingConfig};
use rust_lstm::optimizers::Adam;
use rust_lstm::loss::MSELoss;

fn main() {
    // Create network with dropout
    let network = LSTMNetwork::new(1, 10, 2)
        .with_input_dropout(0.2, true)
        .with_recurrent_dropout(0.3, true);
    
    // Setup trainer
    let mut trainer = create_basic_trainer(
        network,
        MSELoss,
        Adam::new(0.001)
    ).with_config(TrainingConfig {
        epochs: 100,
        clip_gradient: Some(1.0),
        ..Default::default()
    });
    
    // Train (train_data is Vec<(input, target)>)
    trainer.train(&train_data, Some(&validation_data));
}
```

### Bidirectional LSTM

```rust
use rust_lstm::layers::bilstm_network::{BiLSTMNetwork, CombineMode};

// BiLSTM with concatenated outputs (output_size = 2 * hidden_size)
let mut bilstm = BiLSTMNetwork::new_concat(input_size, hidden_size, num_layers);

// Process sequence with both past and future context
let outputs = bilstm.forward_sequence(&sequence);
```

### GRU Networks

```rust
use rust_lstm::models::gru_network::GRUNetwork;

// Create GRU network (alternative to LSTM)
let mut gru = GRUNetwork::new(input_size, hidden_size, num_layers)
    .with_input_dropout(0.2, true)
    .with_recurrent_dropout(0.3, true);

// Forward pass
let (output, _) = gru.forward(&input, &hidden_state);
```

### Learning Rate Scheduling

```rust
use rust_lstm::{create_step_lr_trainer, create_one_cycle_trainer};

// Step decay: reduce LR by 50% every 10 epochs
let mut trainer = create_step_lr_trainer(network, 0.01, 10, 0.5);

// OneCycle policy for modern deep learning
let mut trainer = create_one_cycle_trainer(network, 0.1, 100);
```

## Architecture

- **`layers`**: LSTM and GRU cells (standard, peephole, bidirectional) with dropout
- **`models`**: High-level network architectures (LSTM, BiLSTM, GRU)
- **`training`**: Training utilities with automatic train/eval mode switching
- **`optimizers`**: SGD, Adam, RMSprop with scheduling
- **`loss`**: MSE, MAE, Cross-entropy loss functions
- **`schedulers`**: Learning rate scheduling algorithms

## Examples

Run examples to see the library in action:

```bash
# Basic training workflow
cargo run --example training_example

# GRU vs LSTM comparison
cargo run --example gru_example

# Comprehensive dropout demo
cargo run --example dropout_example

# Bidirectional LSTM
cargo run --example bilstm_example

# Learning rate scheduling
cargo run --example learning_rate_scheduling

# Real-world applications
cargo run --example stock_prediction
cargo run --example weather_prediction
cargo run --example text_classification_bilstm
```

## Advanced Features

### Dropout Types
- **Input Dropout**: Applied to inputs before computing gates
- **Recurrent Dropout**: Applied to hidden states with variational support
- **Output Dropout**: Applied to layer outputs
- **Zoneout**: RNN-specific regularization preserving previous states

### Optimizers
- **SGD**: Stochastic gradient descent with momentum
- **Adam**: Adaptive moment estimation with bias correction
- **RMSprop**: Root mean square propagation

### Loss Functions  
- **MSELoss**: Mean squared error for regression
- **MAELoss**: Mean absolute error for robust regression
- **CrossEntropyLoss**: Numerically stable softmax cross-entropy for classification

### Learning Rate Schedulers
- **StepLR**: Decay by factor every N epochs
- **OneCycleLR**: One cycle policy (warmup + annealing)
- **CosineAnnealingLR**: Smooth cosine oscillation
- **ReduceLROnPlateau**: Reduce when validation loss plateaus

## Testing

```bash
cargo test
```

## Version History

- **v0.3.0**: Bidirectional LSTM networks with flexible combine modes
- **v0.2.0**: Complete training system with BPTT and comprehensive dropout
- **v0.1.0**: Initial LSTM implementation with forward pass

## Contributing

Contributions are welcome! Please submit issues, feature requests, or pull requests.

## License

MIT License - see the LICENSE file for details.
