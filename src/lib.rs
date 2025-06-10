//! # Rust LSTM Library
//! 
//! A complete LSTM implementation with training capabilities, multiple optimizers,
//! and support for various architectures including peephole connections.
//! 
//! ## Core Components
//! 
//! - **LSTM Cells**: Standard and peephole LSTM implementations with full backpropagation
//! - **Networks**: Multi-layer LSTM networks for sequence modeling
//! - **Training**: Complete training system with BPTT, gradient clipping, and validation
//! - **Optimizers**: SGD, Adam, and RMSprop optimizers with adaptive learning rates
//! - **Loss Functions**: MSE, MAE, and Cross-Entropy with numerically stable implementations
//! 
//! ## Quick Start
//! 
//! ```rust
//! use rust_lstm::models::lstm_network::LSTMNetwork;
//! use rust_lstm::training::create_basic_trainer;
//! 
//! // Create a 2-layer LSTM with 10 input features and 20 hidden units
//! let network = LSTMNetwork::new(10, 20, 2);
//! let mut trainer = create_basic_trainer(network, 0.001);
//! 
//! // Train on your data
//! // trainer.train(&train_data, Some(&validation_data));
//! ```

/// Main library module.
pub mod utils;
pub mod layers;
pub mod models;
pub mod loss;
pub mod optimizers;
pub mod training;
pub mod persistence;

// Re-export commonly used items
pub use models::lstm_network::LSTMNetwork;
pub use layers::lstm_cell::LSTMCell;
pub use layers::peephole_lstm_cell::PeepholeLSTMCell;
pub use training::{LSTMTrainer, TrainingConfig};
pub use optimizers::{SGD, Adam, RMSprop};
pub use loss::{MSELoss, MAELoss, CrossEntropyLoss};
pub use persistence::{ModelPersistence, PersistentModel, ModelMetadata, PersistenceError};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_library_integration() {
        let network = models::lstm_network::LSTMNetwork::new(2, 3, 1);
        let input = arr2(&[[1.0], [0.5]]);
        let hx = arr2(&[[0.0], [0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0], [0.0]]);
        
        let (hy, cy) = network.forward(&input, &hx, &cx);
        
        assert_eq!(hy.shape(), &[3, 1]);
        assert_eq!(cy.shape(), &[3, 1]);
    }
}
