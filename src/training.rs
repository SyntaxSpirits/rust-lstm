use ndarray::Array2;
use crate::models::lstm_network::LSTMNetwork;
use crate::loss::{LossFunction, MSELoss};
use crate::optimizers::{Optimizer, SGD};
use std::time::Instant;

/// Configuration for training hyperparameters
pub struct TrainingConfig {
    pub epochs: usize,
    pub print_every: usize,
    pub clip_gradient: Option<f64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            epochs: 100,
            print_every: 10,
            clip_gradient: Some(5.0),
        }
    }
}

/// Training metrics tracked during training
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
    pub time_elapsed: f64,
}

/// Main trainer for LSTM networks with configurable loss and optimizer
pub struct LSTMTrainer<L: LossFunction, O: Optimizer> {
    pub network: LSTMNetwork,
    pub loss_function: L,
    pub optimizer: O,
    pub config: TrainingConfig,
    pub metrics_history: Vec<TrainingMetrics>,
}

impl<L: LossFunction, O: Optimizer> LSTMTrainer<L, O> {
    pub fn new(network: LSTMNetwork, loss_function: L, optimizer: O) -> Self {
        LSTMTrainer {
            network,
            loss_function,
            optimizer,
            config: TrainingConfig::default(),
            metrics_history: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        self.config = config;
        self
    }

    /// Train on a single sequence using backpropagation through time (BPTT)
    pub fn train_sequence(&mut self, inputs: &[Array2<f64>], targets: &[Array2<f64>]) -> f64 {
        if inputs.len() != targets.len() {
            panic!("Inputs and targets must have the same length");
        }

        let (outputs, caches) = self.network.forward_sequence_with_cache(inputs);
        
        let mut total_loss = 0.0;
        let mut total_gradients = self.network.zero_gradients();

        // BPTT: accumulate gradients across all timesteps
        for (i, ((output, _), target)) in outputs.iter().zip(targets.iter()).enumerate().rev() {
            let loss = self.loss_function.compute_loss(output, target);
            total_loss += loss;

            let dhy = self.loss_function.compute_gradient(output, target);
            let dcy = Array2::zeros(output.raw_dim());

            let (step_gradients, _) = self.network.backward(&dhy, &dcy, &caches[i]);

            // Accumulate gradients across timesteps
            for (total_grad, step_grad) in total_gradients.iter_mut().zip(step_gradients.iter()) {
                total_grad.w_ih = &total_grad.w_ih + &step_grad.w_ih;
                total_grad.w_hh = &total_grad.w_hh + &step_grad.w_hh;
                total_grad.b_ih = &total_grad.b_ih + &step_grad.b_ih;
                total_grad.b_hh = &total_grad.b_hh + &step_grad.b_hh;
            }
        }

        // Apply gradient clipping to prevent exploding gradients
        if let Some(clip_value) = self.config.clip_gradient {
            self.clip_gradients(&mut total_gradients, clip_value);
        }

        self.network.update_parameters(&total_gradients, &mut self.optimizer);

        total_loss / inputs.len() as f64
    }

    /// Train for multiple epochs with optional validation
    pub fn train(&mut self, train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                 validation_data: Option<&[(Vec<Array2<f64>>, Vec<Array2<f64>>)]>) {
        
        println!("Starting training for {} epochs...", self.config.epochs);
        
        for epoch in 0..self.config.epochs {
            let start_time = Instant::now();
            let mut epoch_loss = 0.0;

            for (inputs, targets) in train_data {
                let loss = self.train_sequence(inputs, targets);
                epoch_loss += loss;
            }
            epoch_loss /= train_data.len() as f64;

            let validation_loss = if let Some(val_data) = validation_data {
                Some(self.evaluate(val_data))
            } else {
                None
            };

            let time_elapsed = start_time.elapsed().as_secs_f64();

            let metrics = TrainingMetrics {
                epoch,
                train_loss: epoch_loss,
                validation_loss,
                time_elapsed,
            };

            self.metrics_history.push(metrics.clone());

            if epoch % self.config.print_every == 0 {
                if let Some(val_loss) = validation_loss {
                    println!("Epoch {}: Train Loss: {:.6}, Val Loss: {:.6}, Time: {:.2}s", 
                             epoch, epoch_loss, val_loss, time_elapsed);
                } else {
                    println!("Epoch {}: Train Loss: {:.6}, Time: {:.2}s", 
                             epoch, epoch_loss, time_elapsed);
                }
            }
        }

        println!("Training completed!");
    }

    /// Evaluate model performance on validation data
    pub fn evaluate(&self, data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) -> f64 {
        let mut total_loss = 0.0;
        let mut total_samples = 0;

        for (inputs, targets) in data {
            if inputs.len() != targets.len() {
                continue;
            }

            let (outputs, _) = self.network.forward_sequence_with_cache(inputs);
            
            for ((output, _), target) in outputs.iter().zip(targets.iter()) {
                let loss = self.loss_function.compute_loss(output, target);
                total_loss += loss;
                total_samples += 1;
            }
        }

        if total_samples > 0 {
            total_loss / total_samples as f64
        } else {
            0.0
        }
    }

    /// Generate predictions for input sequences
    pub fn predict(&self, inputs: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let (outputs, _) = self.network.forward_sequence_with_cache(inputs);
        outputs.into_iter().map(|(output, _)| output).collect()
    }

    /// Clip gradients by global norm to prevent exploding gradients
    fn clip_gradients(&self, gradients: &mut [crate::layers::lstm_cell::LSTMCellGradients], max_norm: f64) {
        for gradient in gradients.iter_mut() {
            self.clip_gradient_matrix(&mut gradient.w_ih, max_norm);
            self.clip_gradient_matrix(&mut gradient.w_hh, max_norm);
            self.clip_gradient_matrix(&mut gradient.b_ih, max_norm);
            self.clip_gradient_matrix(&mut gradient.b_hh, max_norm);
        }
    }

    fn clip_gradient_matrix(&self, matrix: &mut Array2<f64>, max_norm: f64) {
        let norm = (&*matrix * &*matrix).sum().sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            *matrix = matrix.map(|x| x * scale);
        }
    }

    pub fn get_latest_metrics(&self) -> Option<&TrainingMetrics> {
        self.metrics_history.last()
    }

    pub fn get_metrics_history(&self) -> &[TrainingMetrics] {
        &self.metrics_history
    }
}

/// Create a basic trainer with SGD optimizer and MSE loss
pub fn create_basic_trainer(network: LSTMNetwork, learning_rate: f64) -> LSTMTrainer<MSELoss, SGD> {
    let loss_function = MSELoss;
    let optimizer = SGD::new(learning_rate);
    LSTMTrainer::new(network, loss_function, optimizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_trainer_creation() {
        let network = LSTMNetwork::new(2, 3, 1);
        let trainer = create_basic_trainer(network, 0.01);
        
        assert_eq!(trainer.network.input_size, 2);
        assert_eq!(trainer.network.hidden_size, 3);
        assert_eq!(trainer.network.num_layers, 1);
    }

    #[test]
    fn test_sequence_training() {
        let network = LSTMNetwork::new(2, 3, 1);
        let mut trainer = create_basic_trainer(network, 0.01);
        
        let inputs = vec![
            arr2(&[[1.0], [0.0]]),
            arr2(&[[0.0], [1.0]]),
        ];
        let targets = vec![
            arr2(&[[1.0], [0.0], [0.0]]),
            arr2(&[[0.0], [1.0], [0.0]]),
        ];
        
        let loss = trainer.train_sequence(&inputs, &targets);
        assert!(loss >= 0.0);
    }
} 