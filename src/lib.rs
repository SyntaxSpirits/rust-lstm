#![allow(clippy::type_complexity)]

//! # Rust LSTM Library
//!
//! A complete LSTM implementation with training capabilities, multiple optimizers,
//! dropout regularization, and support for various architectures including peephole
//! connections and bidirectional processing.
//!
//! ## Core Components
//!
//! - **LSTM Cells**: Standard and peephole LSTM implementations with full backpropagation
//! - **Bidirectional LSTM**: Process sequences in both directions with flexible output combination
//! - **Networks**: Multi-layer LSTM networks for sequence modeling
//! - **Training**: Complete training system with BPTT, gradient clipping, and validation
//! - **Optimizers**: SGD, Adam, and RMSprop optimizers with adaptive learning rates
//! - **Loss Functions**: MSE, MAE, and Cross-Entropy with numerically stable implementations
//! - **Dropout**: Input, recurrent, output dropout and zoneout regularization
//!
//! ## Quick Start
//!
//! ```rust
//! use rust_lstm::models::lstm_network::LSTMNetwork;
//! use rust_lstm::training::create_basic_trainer;
//!
//! // Create a 2-layer LSTM with 10 input features and 20 hidden units
//! let mut network = LSTMNetwork::new(10, 20, 2)
//!     .with_input_dropout(0.2, true)     // Variational input dropout
//!     .with_recurrent_dropout(0.3, true) // Variational recurrent dropout
//!     .with_output_dropout(0.1);         // Standard output dropout
//!
//! let mut trainer = create_basic_trainer(network, 0.001);
//!
//! // Train on your data
//! // trainer.train(&train_data, Some(&validation_data));
//! ```

pub mod layers;
pub mod loss;
pub mod models;
pub mod optimizers;
pub mod persistence;
pub mod schedulers;
pub mod text;
pub mod training;
/// Main library module.
pub mod utils;

// Re-export commonly used items
pub use layers::bilstm_network::{BiLSTMNetwork, BiLSTMNetworkCache, CombineMode};
pub use layers::dropout::{Dropout, Zoneout};
pub use layers::gru_cell::{GRUCell, GRUCellCache, GRUCellGradients};
pub use layers::linear::{LinearGradients, LinearLayer};
pub use layers::lstm_cell::{LSTMCell, LSTMCellBatchCache, LSTMCellCache, LSTMCellGradients};
pub use layers::peephole_lstm_cell::PeepholeLSTMCell;
pub use loss::{CrossEntropyLoss, LossFunction, MAELoss, MSELoss};
pub use models::gru_network::{
    GRUNetwork, GRUNetworkCache, LayerDropoutConfig as GRULayerDropoutConfig,
};
pub use models::lstm_network::{
    LSTMNetwork, LSTMNetworkBatchCache, LSTMNetworkCache, LayerDropoutConfig,
};
pub use optimizers::{Adam, RMSprop, ScheduledOptimizer, SGD};
pub use persistence::{ModelMetadata, ModelPersistence, PersistenceError, PersistentModel};
pub use schedulers::{
    AnnealStrategy, ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicalLR,
    CyclicalMode, ExponentialLR, LRScheduleVisualizer, LearningRateScheduler, LinearLR,
    MultiStepLR, OneCycleLR, PolynomialLR, ReduceLROnPlateau, ScaleMode, StepLR, WarmupScheduler,
};
pub use text::{
    argmax, sample_nucleus, sample_top_k, sample_with_temperature, softmax, CharacterEmbedding,
    EmbeddingGradients, TextVocabulary,
};
pub use training::{
    create_adam_batch_trainer, create_basic_batch_trainer, create_basic_trainer,
    create_cosine_annealing_trainer, create_one_cycle_trainer, create_step_lr_trainer,
    EarlyStopper, EarlyStoppingConfig, EarlyStoppingMetric, LSTMBatchTrainer, LSTMTrainer,
    ScheduledLSTMTrainer, TrainingConfig, TrainingMetrics,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_library_integration() {
        let mut network = models::lstm_network::LSTMNetwork::new(2, 3, 1);
        let input = arr2(&[[1.0], [0.5]]);
        let hx = arr2(&[[0.0], [0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0], [0.0]]);

        let (hy, cy) = network.forward(&input, &hx, &cx);

        assert_eq!(hy.shape(), &[3, 1]);
        assert_eq!(cy.shape(), &[3, 1]);
    }

    #[test]
    fn test_library_with_dropout() {
        let mut network = models::lstm_network::LSTMNetwork::new(2, 3, 1)
            .with_input_dropout(0.2, false)
            .with_recurrent_dropout(0.3, true)
            .with_output_dropout(0.1);

        let input = arr2(&[[1.0], [0.5]]);
        let hx = arr2(&[[0.0], [0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0], [0.0]]);

        // Test training mode
        network.train();
        let (hy_train, cy_train) = network.forward(&input, &hx, &cx);

        // Test evaluation mode
        network.eval();
        let (hy_eval, cy_eval) = network.forward(&input, &hx, &cx);

        assert_eq!(hy_train.shape(), &[3, 1]);
        assert_eq!(cy_train.shape(), &[3, 1]);
        assert_eq!(hy_eval.shape(), &[3, 1]);
        assert_eq!(cy_eval.shape(), &[3, 1]);
    }
}
