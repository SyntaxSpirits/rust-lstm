use ndarray::Array2;
use crate::models::lstm_network::LSTMNetwork;
use crate::loss::{LossFunction, MSELoss};
use crate::optimizers::{Optimizer, SGD, ScheduledOptimizer};
use crate::schedulers::LearningRateScheduler;
use crate::persistence::SerializableLSTMNetwork;
use std::time::Instant;

/// Configuration for training hyperparameters
pub struct TrainingConfig {
    pub epochs: usize,
    pub print_every: usize,
    pub clip_gradient: Option<f64>,
    pub log_lr_changes: bool,
    pub early_stopping: Option<EarlyStoppingConfig>,
}

/// Configuration for early stopping
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of epochs with no improvement after which training will be stopped
    pub patience: usize,
    /// Minimum change in the monitored quantity to qualify as an improvement
    pub min_delta: f64,
    /// Whether to restore the best weights when early stopping triggers
    pub restore_best_weights: bool,
    /// Metric to monitor for early stopping ('val_loss' or 'train_loss')
    pub monitor: EarlyStoppingMetric,
}

/// Metric to monitor for early stopping
#[derive(Debug, Clone, PartialEq)]
pub enum EarlyStoppingMetric {
    ValidationLoss,
    TrainLoss,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        EarlyStoppingConfig {
            patience: 10,
            min_delta: 1e-4,
            restore_best_weights: true,
            monitor: EarlyStoppingMetric::ValidationLoss,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            epochs: 100,
            print_every: 10,
            clip_gradient: Some(5.0),
            log_lr_changes: true,
            early_stopping: None,
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
    pub learning_rate: f64,
}

/// Early stopping state tracker
#[derive(Debug, Clone)]
pub struct EarlyStopper {
    config: EarlyStoppingConfig,
    best_score: f64,
    wait_count: usize,
    stopped_epoch: Option<usize>,
    best_weights: Option<SerializableLSTMNetwork>, // Serialized network weights
}

impl EarlyStopper {
    pub fn new(config: EarlyStoppingConfig) -> Self {
        EarlyStopper {
            config,
            best_score: f64::INFINITY,
            wait_count: 0,
            stopped_epoch: None,
            best_weights: None,
        }
    }

    /// Check if training should stop based on current metrics
    /// Returns (should_stop, is_best_score)
    pub fn should_stop(&mut self, current_metrics: &TrainingMetrics, network: &LSTMNetwork) -> (bool, bool) {
        let current_score = match self.config.monitor {
            EarlyStoppingMetric::ValidationLoss => {
                match current_metrics.validation_loss {
                    Some(val_loss) => val_loss,
                    None => {
                        // If validation loss is not available, fall back to train loss
                        current_metrics.train_loss
                    }
                }
            }
            EarlyStoppingMetric::TrainLoss => current_metrics.train_loss,
        };

        let is_improvement = current_score < self.best_score - self.config.min_delta;
        
        if is_improvement {
            self.best_score = current_score;
            self.wait_count = 0;
            
            // Save best weights if restore_best_weights is enabled
            if self.config.restore_best_weights {
                self.best_weights = Some(network.into());
            }
            
            (false, true)
        } else {
            self.wait_count += 1;
            
            if self.wait_count >= self.config.patience {
                self.stopped_epoch = Some(current_metrics.epoch);
                (true, false)
            } else {
                (false, false)
            }
        }
    }

    /// Get the epoch where training was stopped
    pub fn stopped_epoch(&self) -> Option<usize> {
        self.stopped_epoch
    }

    /// Get the best score achieved
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Restore the best weights to the network if available
    pub fn restore_best_weights(&self, network: &mut LSTMNetwork) -> Result<(), String> {
        if let Some(ref weights) = self.best_weights {
            *network = weights.clone().into();
            Ok(())
        } else {
            Err("No best weights available to restore".to_string())
        }
    }
}

/// Main trainer for LSTM networks with configurable loss and optimizer
pub struct LSTMTrainer<L: LossFunction, O: Optimizer> {
    pub network: LSTMNetwork,
    pub loss_function: L,
    pub optimizer: O,
    pub config: TrainingConfig,
    pub metrics_history: Vec<TrainingMetrics>,
    early_stopper: Option<EarlyStopper>,
}

impl<L: LossFunction, O: Optimizer> LSTMTrainer<L, O> {
    pub fn new(network: LSTMNetwork, loss_function: L, optimizer: O) -> Self {
        LSTMTrainer {
            network,
            loss_function,
            optimizer,
            config: TrainingConfig::default(),
            metrics_history: Vec::new(),
            early_stopper: None,
        }
    }

    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        // Initialize early stopper if early stopping is configured
        self.early_stopper = config.early_stopping.as_ref().map(|es_config| {
            EarlyStopper::new(es_config.clone())
        });
        self.config = config;
        self
    }

    /// Train on a single sequence using backpropagation through time (BPTT)
    pub fn train_sequence(&mut self, inputs: &[Array2<f64>], targets: &[Array2<f64>]) -> f64 {
        if inputs.len() != targets.len() {
            panic!("Inputs and targets must have the same length");
        }

        self.network.train();

        let (outputs, caches) = self.network.forward_sequence_with_cache(inputs);
        
        let mut total_loss = 0.0;
        let mut total_gradients = self.network.zero_gradients();

        for (i, ((output, _), target)) in outputs.iter().zip(targets.iter()).enumerate().rev() {
            let loss = self.loss_function.compute_loss(output, target);
            total_loss += loss;

            let dhy = self.loss_function.compute_gradient(output, target);
            let dcy = Array2::zeros(output.raw_dim());

            let (step_gradients, _) = self.network.backward(&dhy, &dcy, &caches[i]);

            for (total_grad, step_grad) in total_gradients.iter_mut().zip(step_gradients.iter()) {
                total_grad.w_ih = &total_grad.w_ih + &step_grad.w_ih;
                total_grad.w_hh = &total_grad.w_hh + &step_grad.w_hh;
                total_grad.b_ih = &total_grad.b_ih + &step_grad.b_ih;
                total_grad.b_hh = &total_grad.b_hh + &step_grad.b_hh;
            }
        }

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

            // Training phase
            self.network.train();
            for (inputs, targets) in train_data {
                let loss = self.train_sequence(inputs, targets);
                epoch_loss += loss;
            }
            epoch_loss /= train_data.len() as f64;

            let validation_loss = if let Some(val_data) = validation_data {
                self.network.eval();
                Some(self.evaluate(val_data))
            } else {
                None
            };

            let time_elapsed = start_time.elapsed().as_secs_f64();

            let current_lr = self.optimizer.get_learning_rate();
            let metrics = TrainingMetrics {
                epoch,
                train_loss: epoch_loss,
                validation_loss,
                time_elapsed,
                learning_rate: current_lr,
            };

            self.metrics_history.push(metrics.clone());

            // Check early stopping
            let mut should_stop = false;
            let mut is_best = false;
            if let Some(ref mut early_stopper) = self.early_stopper {
                let (stop, best) = early_stopper.should_stop(&metrics, &self.network);
                should_stop = stop;
                is_best = best;
            }

            if epoch % self.config.print_every == 0 {
                let best_indicator = if is_best { " *" } else { "" };
                if let Some(val_loss) = validation_loss {
                    println!("Epoch {}: Train Loss: {:.6}, Val Loss: {:.6}, LR: {:.2e}, Time: {:.2}s{}", 
                             epoch, epoch_loss, val_loss, current_lr, time_elapsed, best_indicator);
                } else {
                    println!("Epoch {}: Train Loss: {:.6}, LR: {:.2e}, Time: {:.2}s{}", 
                             epoch, epoch_loss, current_lr, time_elapsed, best_indicator);
                }
            }

            if should_stop {
                let stopped_epoch = self.early_stopper.as_ref().unwrap().stopped_epoch().unwrap();
                let best_score = self.early_stopper.as_ref().unwrap().best_score();
                println!("Early stopping triggered at epoch {} (best score: {:.6})", stopped_epoch, best_score);
                
                // Restore best weights if configured
                if let Some(ref early_stopper) = self.early_stopper {
                    if let Err(e) = early_stopper.restore_best_weights(&mut self.network) {
                        println!("Warning: Could not restore best weights: {}", e);
                    } else {
                        println!("Restored best weights from epoch with score {:.6}", best_score);
                    }
                }
                break;
            }
        }

        println!("Training completed!");
    }

    /// Evaluate model performance on validation data
    pub fn evaluate(&mut self, data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) -> f64 {
        self.network.eval();
        
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
    pub fn predict(&mut self, inputs: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.network.eval();
        
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

    /// Set network to training mode
    pub fn set_training_mode(&mut self, training: bool) {
        if training {
            self.network.train();
        } else {
            self.network.eval();
        }
    }
}

/// Specialized trainer for scheduled optimizers that automatically steps the scheduler
pub struct ScheduledLSTMTrainer<L: LossFunction, O: Optimizer, S: LearningRateScheduler> {
    pub network: LSTMNetwork,
    pub loss_function: L,
    pub optimizer: ScheduledOptimizer<O, S>,
    pub config: TrainingConfig,
    pub metrics_history: Vec<TrainingMetrics>,
    early_stopper: Option<EarlyStopper>,
}

impl<L: LossFunction, O: Optimizer, S: LearningRateScheduler> ScheduledLSTMTrainer<L, O, S> {
    pub fn new(network: LSTMNetwork, loss_function: L, optimizer: ScheduledOptimizer<O, S>) -> Self {
        ScheduledLSTMTrainer {
            network,
            loss_function,
            optimizer,
            config: TrainingConfig::default(),
            metrics_history: Vec::new(),
            early_stopper: None,
        }
    }

    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        // Initialize early stopper if early stopping is configured
        self.early_stopper = config.early_stopping.as_ref().map(|es_config| {
            EarlyStopper::new(es_config.clone())
        });
        self.config = config;
        self
    }

    /// Train on a single sequence using backpropagation through time (BPTT)
    pub fn train_sequence(&mut self, inputs: &[Array2<f64>], targets: &[Array2<f64>]) -> f64 {
        if inputs.len() != targets.len() {
            panic!("Inputs and targets must have the same length");
        }

        self.network.train();

        let (outputs, caches) = self.network.forward_sequence_with_cache(inputs);
        
        let mut total_loss = 0.0;
        let mut total_gradients = self.network.zero_gradients();

        for (i, ((output, _), target)) in outputs.iter().zip(targets.iter()).enumerate().rev() {
            let loss = self.loss_function.compute_loss(output, target);
            total_loss += loss;

            let dhy = self.loss_function.compute_gradient(output, target);
            let dcy = Array2::zeros(output.raw_dim());

            let (step_gradients, _) = self.network.backward(&dhy, &dcy, &caches[i]);

            for (total_grad, step_grad) in total_gradients.iter_mut().zip(step_gradients.iter()) {
                total_grad.w_ih = &total_grad.w_ih + &step_grad.w_ih;
                total_grad.w_hh = &total_grad.w_hh + &step_grad.w_hh;
                total_grad.b_ih = &total_grad.b_ih + &step_grad.b_ih;
                total_grad.b_hh = &total_grad.b_hh + &step_grad.b_hh;
            }
        }

        if let Some(clip_value) = self.config.clip_gradient {
            self.clip_gradients(&mut total_gradients, clip_value);
        }

        self.network.update_parameters(&total_gradients, &mut self.optimizer);

        total_loss / inputs.len() as f64
    }

    /// Train for multiple epochs with automatic scheduler stepping
    pub fn train(&mut self, train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                 validation_data: Option<&[(Vec<Array2<f64>>, Vec<Array2<f64>>)]>) {
        
        println!("Starting training for {} epochs with {} scheduler...", 
                 self.config.epochs, self.optimizer.scheduler_name());
        
        for epoch in 0..self.config.epochs {
            let start_time = Instant::now();
            let mut epoch_loss = 0.0;

            // Training phase
            self.network.train();
            for (inputs, targets) in train_data {
                let loss = self.train_sequence(inputs, targets);
                epoch_loss += loss;
            }
            epoch_loss /= train_data.len() as f64;

            let validation_loss = if let Some(val_data) = validation_data {
                self.network.eval();
                Some(self.evaluate(val_data))
            } else {
                None
            };

            // Step the scheduler at the end of each epoch
            let prev_lr = self.optimizer.get_learning_rate();
            if let Some(val_loss) = validation_loss {
                self.optimizer.step_with_val_loss(val_loss);
            } else {
                self.optimizer.step();
            }
            let new_lr = self.optimizer.get_learning_rate();

            // Log learning rate changes if enabled
            if self.config.log_lr_changes && (new_lr - prev_lr).abs() > 1e-10 {
                println!("Learning rate changed from {:.2e} to {:.2e}", prev_lr, new_lr);
            }

            let time_elapsed = start_time.elapsed().as_secs_f64();

            let metrics = TrainingMetrics {
                epoch,
                train_loss: epoch_loss,
                validation_loss,
                time_elapsed,
                learning_rate: new_lr,
            };

            self.metrics_history.push(metrics.clone());

            // Check early stopping
            let mut should_stop = false;
            let mut is_best = false;
            if let Some(ref mut early_stopper) = self.early_stopper {
                let (stop, best) = early_stopper.should_stop(&metrics, &self.network);
                should_stop = stop;
                is_best = best;
            }

            if epoch % self.config.print_every == 0 {
                let best_indicator = if is_best { " *" } else { "" };
                if let Some(val_loss) = validation_loss {
                    println!("Epoch {}: Train Loss: {:.6}, Val Loss: {:.6}, LR: {:.2e}, Time: {:.2}s{}", 
                             epoch, epoch_loss, val_loss, new_lr, time_elapsed, best_indicator);
                } else {
                    println!("Epoch {}: Train Loss: {:.6}, LR: {:.2e}, Time: {:.2}s{}", 
                             epoch, epoch_loss, new_lr, time_elapsed, best_indicator);
                }
            }

            if should_stop {
                let stopped_epoch = self.early_stopper.as_ref().unwrap().stopped_epoch().unwrap();
                let best_score = self.early_stopper.as_ref().unwrap().best_score();
                println!("Early stopping triggered at epoch {} (best score: {:.6})", stopped_epoch, best_score);
                
                // Restore best weights if configured
                if let Some(ref early_stopper) = self.early_stopper {
                    if let Err(e) = early_stopper.restore_best_weights(&mut self.network) {
                        println!("Warning: Could not restore best weights: {}", e);
                    } else {
                        println!("Restored best weights from epoch with score {:.6}", best_score);
                    }
                }
                break;
            }
        }

        println!("Training completed!");
    }

    /// Evaluate model performance on validation data
    pub fn evaluate(&mut self, data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) -> f64 {
        self.network.eval();
        
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
    pub fn predict(&mut self, inputs: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.network.eval();
        
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

    /// Set network to training mode
    pub fn set_training_mode(&mut self, training: bool) {
        if training {
            self.network.train();
        } else {
            self.network.eval();
        }
    }

    /// Get the current learning rate
    pub fn get_current_lr(&self) -> f64 {
        self.optimizer.get_current_lr()
    }

    /// Get the current epoch from the scheduler
    pub fn get_current_epoch(&self) -> usize {
        self.optimizer.get_current_epoch()
    }

    /// Reset the optimizer and scheduler
    pub fn reset_optimizer(&mut self) {
        self.optimizer.reset();
    }
}

/// Batch trainer for LSTM networks with configurable loss and optimizer
/// Processes multiple sequences simultaneously for improved performance
pub struct LSTMBatchTrainer<L: LossFunction, O: Optimizer> {
    pub network: LSTMNetwork,
    pub loss_function: L,
    pub optimizer: O,
    pub config: TrainingConfig,
    pub metrics_history: Vec<TrainingMetrics>,
    early_stopper: Option<EarlyStopper>,
}

impl<L: LossFunction, O: Optimizer> LSTMBatchTrainer<L, O> {
    pub fn new(network: LSTMNetwork, loss_function: L, optimizer: O) -> Self {
        LSTMBatchTrainer {
            network,
            loss_function,
            optimizer,
            config: TrainingConfig::default(),
            metrics_history: Vec::new(),
            early_stopper: None,
        }
    }

    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        // Initialize early stopper if early stopping is configured
        self.early_stopper = config.early_stopping.as_ref().map(|es_config| {
            EarlyStopper::new(es_config.clone())
        });
        self.config = config;
        self
    }

    /// Train on a batch of sequences using batch processing
    /// 
    /// # Arguments
    /// * `batch_inputs` - Vector of input sequences, each sequence is Vec<Array2<f64>>
    /// * `batch_targets` - Vector of target sequences, each sequence is Vec<Array2<f64>>
    /// 
    /// # Returns
    /// * Average loss across the batch
    pub fn train_batch(&mut self, batch_inputs: &[Vec<Array2<f64>>], batch_targets: &[Vec<Array2<f64>>]) -> f64 {
        assert_eq!(batch_inputs.len(), batch_targets.len(), "Batch inputs and targets must have same length");
        
        if batch_inputs.is_empty() {
            return 0.0;
        }

        self.network.train();

        // Find maximum sequence length for padding
        let max_seq_len = batch_inputs.iter().map(|seq| seq.len()).max().unwrap_or(0);
        let batch_size = batch_inputs.len();

        let mut total_loss = 0.0;
        let mut total_gradients = self.network.zero_gradients();
        let mut valid_steps = 0;

        // Initialize batch states
        let mut batch_hx = Array2::zeros((self.network.hidden_size, batch_size));
        let mut batch_cx = Array2::zeros((self.network.hidden_size, batch_size));

        // Process each time step
        for t in 0..max_seq_len {
            // Prepare batch input and targets for current time step
            let mut batch_input = Array2::zeros((self.network.input_size, batch_size));
            let mut batch_target = Array2::zeros((self.network.hidden_size, batch_size));
            let mut active_sequences = Vec::new();

            // Collect active sequences for this time step
            for (batch_idx, (input_seq, target_seq)) in batch_inputs.iter().zip(batch_targets.iter()).enumerate() {
                if t < input_seq.len() && t < target_seq.len() {
                    batch_input.column_mut(batch_idx).assign(&input_seq[t].column(0));
                    batch_target.column_mut(batch_idx).assign(&target_seq[t].column(0));
                    active_sequences.push(batch_idx);
                }
            }

            if active_sequences.is_empty() {
                break;
            }

            // Forward pass with caching for active sequences
            let (new_batch_hx, new_batch_cx, cache) = self.network.forward_batch_with_cache(&batch_input, &batch_hx, &batch_cx);

            // Compute loss only for active sequences
            let active_predictions = if active_sequences.len() == batch_size {
                new_batch_hx.clone()
            } else {
                let mut active_preds = Array2::zeros((self.network.hidden_size, active_sequences.len()));
                for (idx, &batch_idx) in active_sequences.iter().enumerate() {
                    active_preds.column_mut(idx).assign(&new_batch_hx.column(batch_idx));
                }
                active_preds
            };

            let active_targets = if active_sequences.len() == batch_size {
                batch_target.clone()
            } else {
                let mut active_targs = Array2::zeros((self.network.hidden_size, active_sequences.len()));
                for (idx, &batch_idx) in active_sequences.iter().enumerate() {
                    active_targs.column_mut(idx).assign(&batch_target.column(batch_idx));
                }
                active_targs
            };

            let step_loss = self.loss_function.compute_batch_loss(&active_predictions, &active_targets);
            total_loss += step_loss;
            valid_steps += 1;

            // Compute gradients
            let dhy = self.loss_function.compute_batch_gradient(&active_predictions, &active_targets);
            let _dcy = Array2::<f64>::zeros(dhy.raw_dim());

            // Expand gradients back to full batch size if needed
            let full_dhy = if active_sequences.len() == batch_size {
                dhy
            } else {
                let mut full_grad = Array2::zeros((self.network.hidden_size, batch_size));
                for (idx, &batch_idx) in active_sequences.iter().enumerate() {
                    full_grad.column_mut(batch_idx).assign(&dhy.column(idx));
                }
                full_grad
            };

            let full_dcy = Array2::<f64>::zeros(full_dhy.raw_dim());

            // Backward pass
            let (step_gradients, _) = self.network.backward_batch(&full_dhy, &full_dcy, &cache);

            // Accumulate gradients
            for (total_grad, step_grad) in total_gradients.iter_mut().zip(step_gradients.iter()) {
                total_grad.w_ih = &total_grad.w_ih + &step_grad.w_ih;
                total_grad.w_hh = &total_grad.w_hh + &step_grad.w_hh;
                total_grad.b_ih = &total_grad.b_ih + &step_grad.b_ih;
                total_grad.b_hh = &total_grad.b_hh + &step_grad.b_hh;
            }

            // Update states
            batch_hx = new_batch_hx;
            batch_cx = new_batch_cx;
        }

        // Apply gradient clipping
        if let Some(clip_value) = self.config.clip_gradient {
            self.clip_gradients(&mut total_gradients, clip_value);
        }

        // Update parameters
        self.network.update_parameters(&total_gradients, &mut self.optimizer);

        if valid_steps > 0 {
            total_loss / valid_steps as f64
        } else {
            0.0
        }
    }

    /// Train for multiple epochs with batch processing
    /// 
    /// # Arguments
    /// * `train_data` - Vector of (input_sequences, target_sequences) tuples for training
    /// * `validation_data` - Optional validation data
    /// * `batch_size` - Number of sequences to process in each batch
    pub fn train(&mut self, 
                 train_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], 
                 validation_data: Option<&[(Vec<Array2<f64>>, Vec<Array2<f64>>)]>,
                 batch_size: usize) {
        
        println!("Starting batch training for {} epochs with batch size {}...", 
                 self.config.epochs, batch_size);
        
        for epoch in 0..self.config.epochs {
            let start_time = Instant::now();
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Create batches
            for batch_start in (0..train_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(train_data.len());
                let batch = &train_data[batch_start..batch_end];
                
                let batch_inputs: Vec<_> = batch.iter().map(|(inputs, _)| inputs.clone()).collect();
                let batch_targets: Vec<_> = batch.iter().map(|(_, targets)| targets.clone()).collect();
                
                let batch_loss = self.train_batch(&batch_inputs, &batch_targets);
                epoch_loss += batch_loss;
                num_batches += 1;
            }

            epoch_loss /= num_batches as f64;

            // Validation
            let validation_loss = if let Some(val_data) = validation_data {
                self.network.eval();
                Some(self.evaluate_batch(val_data, batch_size))
            } else {
                None
            };

            let time_elapsed = start_time.elapsed().as_secs_f64();
            let current_lr = self.optimizer.get_learning_rate();

            let metrics = TrainingMetrics {
                epoch,
                train_loss: epoch_loss,
                validation_loss,
                time_elapsed,
                learning_rate: current_lr,
            };

            self.metrics_history.push(metrics.clone());

            // Check early stopping
            let mut should_stop = false;
            let mut is_best = false;
            if let Some(ref mut early_stopper) = self.early_stopper {
                let (stop, best) = early_stopper.should_stop(&metrics, &self.network);
                should_stop = stop;
                is_best = best;
            }

            if epoch % self.config.print_every == 0 {
                let best_indicator = if is_best { " *" } else { "" };
                if let Some(val_loss) = validation_loss {
                    println!("Epoch {}: Train Loss: {:.6}, Val Loss: {:.6}, LR: {:.2e}, Time: {:.2}s, Batches: {}{}", 
                             epoch, epoch_loss, val_loss, current_lr, time_elapsed, num_batches, best_indicator);
                } else {
                    println!("Epoch {}: Train Loss: {:.6}, LR: {:.2e}, Time: {:.2}s, Batches: {}{}", 
                             epoch, epoch_loss, current_lr, time_elapsed, num_batches, best_indicator);
                }
            }

            if should_stop {
                let stopped_epoch = self.early_stopper.as_ref().unwrap().stopped_epoch().unwrap();
                let best_score = self.early_stopper.as_ref().unwrap().best_score();
                println!("Early stopping triggered at epoch {} (best score: {:.6})", stopped_epoch, best_score);
                
                // Restore best weights if configured
                if let Some(ref early_stopper) = self.early_stopper {
                    if let Err(e) = early_stopper.restore_best_weights(&mut self.network) {
                        println!("Warning: Could not restore best weights: {}", e);
                    } else {
                        println!("Restored best weights from epoch with score {:.6}", best_score);
                    }
                }
                break;
            }
        }

        println!("Batch training completed!");
    }

    /// Evaluate model performance using batch processing
    pub fn evaluate_batch(&mut self, data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)], batch_size: usize) -> f64 {
        self.network.eval();
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch_start in (0..data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(data.len());
            let batch = &data[batch_start..batch_end];
            
            let batch_inputs: Vec<_> = batch.iter().map(|(inputs, _)| inputs.clone()).collect();
            let batch_targets: Vec<_> = batch.iter().map(|(_, targets)| targets.clone()).collect();
            
            // Process batch and compute loss (simplified evaluation)
            let batch_outputs = self.network.forward_batch_sequences(&batch_inputs);
            
            let mut batch_loss = 0.0;
            let mut valid_samples = 0;
            
            for (outputs, targets) in batch_outputs.iter().zip(batch_targets.iter()) {
                for ((output, _), target) in outputs.iter().zip(targets.iter()) {
                    let loss = self.loss_function.compute_loss(output, target);
                    batch_loss += loss;
                    valid_samples += 1;
                }
            }
            
            if valid_samples > 0 {
                total_loss += batch_loss / valid_samples as f64;
                num_batches += 1;
            }
        }

        if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        }
    }

    /// Generate predictions using batch processing
    pub fn predict_batch(&mut self, inputs: &[Vec<Array2<f64>>]) -> Vec<Vec<Array2<f64>>> {
        self.network.eval();
        
        let batch_outputs = self.network.forward_batch_sequences(inputs);
        batch_outputs.into_iter()
            .map(|sequence_outputs| sequence_outputs.into_iter().map(|(output, _)| output).collect())
            .collect()
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

    pub fn set_training_mode(&mut self, training: bool) {
        if training {
            self.network.train();
        } else {
            self.network.eval();
        }
    }
}

/// Create a basic trainer with SGD optimizer and MSE loss
pub fn create_basic_trainer(network: LSTMNetwork, learning_rate: f64) -> LSTMTrainer<MSELoss, SGD> {
    let loss_function = MSELoss;
    let optimizer = SGD::new(learning_rate);
    LSTMTrainer::new(network, loss_function, optimizer)
}

/// Create a scheduled trainer with SGD and StepLR scheduler
pub fn create_step_lr_trainer(
    network: LSTMNetwork, 
    learning_rate: f64, 
    step_size: usize, 
    gamma: f64
) -> ScheduledLSTMTrainer<MSELoss, SGD, crate::schedulers::StepLR> {
    let loss_function = MSELoss;
    let optimizer = ScheduledOptimizer::step_lr(SGD::new(learning_rate), learning_rate, step_size, gamma);
    ScheduledLSTMTrainer::new(network, loss_function, optimizer)
}

/// Create a scheduled trainer with Adam and OneCycleLR scheduler  
pub fn create_one_cycle_trainer(
    network: LSTMNetwork,
    max_lr: f64,
    total_steps: usize
) -> ScheduledLSTMTrainer<MSELoss, crate::optimizers::Adam, crate::schedulers::OneCycleLR> {
    let loss_function = MSELoss;
    let optimizer = ScheduledOptimizer::one_cycle(
        crate::optimizers::Adam::new(max_lr), 
        max_lr, 
        total_steps
    );
    ScheduledLSTMTrainer::new(network, loss_function, optimizer)
}

/// Create a scheduled trainer with Adam and CosineAnnealingLR scheduler
pub fn create_cosine_annealing_trainer(
    network: LSTMNetwork,
    learning_rate: f64,
    t_max: usize,
    eta_min: f64
) -> ScheduledLSTMTrainer<MSELoss, crate::optimizers::Adam, crate::schedulers::CosineAnnealingLR> {
    let loss_function = MSELoss;
    let optimizer = crate::optimizers::Adam::new(learning_rate);
    let scheduler = crate::schedulers::CosineAnnealingLR::new(t_max, eta_min);
    let scheduled_optimizer = crate::optimizers::ScheduledOptimizer::new(optimizer, scheduler, learning_rate);
    
    ScheduledLSTMTrainer::new(network, loss_function, scheduled_optimizer)
}

/// Create a basic batch trainer with SGD optimizer and MSE loss
pub fn create_basic_batch_trainer(network: LSTMNetwork, learning_rate: f64) -> LSTMBatchTrainer<MSELoss, SGD> {
    let loss_function = MSELoss;
    let optimizer = SGD::new(learning_rate);
    LSTMBatchTrainer::new(network, loss_function, optimizer)
}

/// Create a batch trainer with Adam optimizer and MSE loss
pub fn create_adam_batch_trainer(network: LSTMNetwork, learning_rate: f64) -> LSTMBatchTrainer<MSELoss, crate::optimizers::Adam> {
    let loss_function = MSELoss;
    let optimizer = crate::optimizers::Adam::new(learning_rate);
    LSTMBatchTrainer::new(network, loss_function, optimizer)
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