use ndarray::Array2;
use crate::layers::gru_cell::{GRUCell, GRUCellGradients, GRUCellCache};
use crate::optimizers::Optimizer;

/// Cache for GRU network forward pass
#[derive(Clone)]
pub struct GRUNetworkCache {
    pub caches: Vec<GRUCellCache>,
}

/// Configuration for layer-specific dropout settings
#[derive(Clone)]
pub struct LayerDropoutConfig {
    pub input_dropout_rate: f64,
    pub input_variational: bool,
    pub recurrent_dropout_rate: f64,
    pub recurrent_variational: bool,
    pub output_dropout_rate: f64,
}

impl LayerDropoutConfig {
    pub fn new() -> Self {
        LayerDropoutConfig {
            input_dropout_rate: 0.0,
            input_variational: false,
            recurrent_dropout_rate: 0.0,
            recurrent_variational: false,
            output_dropout_rate: 0.0,
        }
    }

    pub fn with_input_dropout(mut self, rate: f64, variational: bool) -> Self {
        self.input_dropout_rate = rate;
        self.input_variational = variational;
        self
    }

    pub fn with_recurrent_dropout(mut self, rate: f64, variational: bool) -> Self {
        self.recurrent_dropout_rate = rate;
        self.recurrent_variational = variational;
        self
    }

    pub fn with_output_dropout(mut self, rate: f64) -> Self {
        self.output_dropout_rate = rate;
        self
    }
}

/// Multi-layer GRU network for sequence modeling
#[derive(Clone)]
pub struct GRUNetwork {
    cells: Vec<GRUCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub is_training: bool,
}

impl GRUNetwork {
    /// Creates a new multi-layer GRU network
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::new();
        
        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            cells.push(GRUCell::new(layer_input_size, hidden_size));
        }
        
        GRUNetwork {
            cells,
            input_size,
            hidden_size,
            num_layers,
            is_training: true,
        }
    }

    /// Apply uniform dropout across all layers
    pub fn with_input_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in &mut self.cells {
            *cell = cell.clone().with_input_dropout(dropout_rate, variational);
        }
        self
    }

    pub fn with_recurrent_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in &mut self.cells {
            *cell = cell.clone().with_recurrent_dropout(dropout_rate, variational);
        }
        self
    }

    pub fn with_output_dropout(mut self, dropout_rate: f64) -> Self {
        // Apply output dropout to all layers except the last
        for (i, cell) in self.cells.iter_mut().enumerate() {
            if i < self.num_layers - 1 {
                *cell = cell.clone().with_output_dropout(dropout_rate);
            }
        }
        self
    }

    /// Apply layer-specific dropout configuration
    pub fn with_layer_dropout(mut self, configs: Vec<LayerDropoutConfig>) -> Self {
        if configs.len() != self.num_layers {
            panic!("Number of dropout configs must match number of layers");
        }

        for (i, config) in configs.into_iter().enumerate() {
            if config.input_dropout_rate > 0.0 {
                self.cells[i] = self.cells[i].clone()
                    .with_input_dropout(config.input_dropout_rate, config.input_variational);
            }
            if config.recurrent_dropout_rate > 0.0 {
                self.cells[i] = self.cells[i].clone()
                    .with_recurrent_dropout(config.recurrent_dropout_rate, config.recurrent_variational);
            }
            if config.output_dropout_rate > 0.0 && i < self.num_layers - 1 {
                self.cells[i] = self.cells[i].clone()
                    .with_output_dropout(config.output_dropout_rate);
            }
        }
        self
    }

    pub fn train(&mut self) {
        self.is_training = true;
        for cell in &mut self.cells {
            cell.train();
        }
    }

    pub fn eval(&mut self) {
        self.is_training = false;
        for cell in &mut self.cells {
            cell.eval();
        }
    }

    /// Forward pass for a single time step
    pub fn forward(&mut self, input: &Array2<f64>, hx: &[Array2<f64>]) -> Vec<Array2<f64>> {
        if hx.len() != self.num_layers {
            panic!("Number of hidden states must match number of layers");
        }

        let mut layer_input = input.clone();
        let mut outputs = Vec::new();

        for (i, cell) in self.cells.iter_mut().enumerate() {
            let hy = cell.forward(&layer_input, &hx[i]);
            outputs.push(hy.clone());
            layer_input = hy;
        }

        outputs
    }

    /// Forward pass for a sequence with caching for training
    pub fn forward_sequence_with_cache(&mut self, sequence: &[Array2<f64>]) -> (Vec<(Array2<f64>, Vec<Array2<f64>>)>, Vec<GRUNetworkCache>) {
        let mut all_outputs = Vec::new();
        let mut all_caches = Vec::new();

        // Initialize hidden states for all layers
        let mut hidden_states: Vec<Array2<f64>> = (0..self.num_layers)
            .map(|_| Array2::zeros((self.hidden_size, 1)))
            .collect();

        for input in sequence {
            let mut layer_input = input.clone();
            let mut step_outputs = Vec::new();
            let mut step_caches = Vec::new();

            for (i, cell) in self.cells.iter_mut().enumerate() {
                let (hy, cache) = cell.forward_with_cache(&layer_input, &hidden_states[i]);
                
                hidden_states[i] = hy.clone();
                step_outputs.push(hy.clone());
                step_caches.push(cache);
                layer_input = hy;
            }

            // The final output is from the last layer
            let final_output = step_outputs.last().unwrap().clone();
            all_outputs.push((final_output, step_outputs));
            all_caches.push(GRUNetworkCache { caches: step_caches });
        }

        (all_outputs, all_caches)
    }

    /// Backward pass for training
    pub fn backward(&self, dhy: &Array2<f64>, cache: &GRUNetworkCache) -> (Vec<GRUCellGradients>, Array2<f64>) {
        let mut gradients = Vec::new();
        let mut dhx = dhy.clone();

        // Backward through layers in reverse order
        for (i, cell) in self.cells.iter().enumerate().rev() {
            let (cell_gradients, _, dhx_prev) = cell.backward(&dhx, &cache.caches[i]);
            gradients.insert(0, cell_gradients);
            dhx = dhx_prev;
        }

        (gradients, dhx)
    }

    /// Update parameters using optimizer
    pub fn update_parameters<O: Optimizer>(&mut self, gradients: &[GRUCellGradients], optimizer: &mut O) {
        for (i, (cell, grad)) in self.cells.iter_mut().zip(gradients.iter()).enumerate() {
            cell.update_parameters(grad, optimizer, &format!("layer_{}", i));
        }
    }

    /// Initialize zero gradients for all layers
    pub fn zero_gradients(&self) -> Vec<GRUCellGradients> {
        self.cells.iter().map(|cell| cell.zero_gradients()).collect()
    }

    /// Get references to cells for inspection
    pub fn get_cells(&self) -> &[GRUCell] {
        &self.cells
    }

    /// Get mutable references to cells
    pub fn get_cells_mut(&mut self) -> &mut [GRUCell] {
        &mut self.cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_gru_network_creation() {
        let network = GRUNetwork::new(3, 5, 2);
        assert_eq!(network.input_size, 3);
        assert_eq!(network.hidden_size, 5);
        assert_eq!(network.num_layers, 2);
        assert_eq!(network.cells.len(), 2);
    }

    #[test]
    fn test_gru_network_forward() {
        let mut network = GRUNetwork::new(2, 3, 2);
        let input = arr2(&[[1.0], [0.5]]);
        let hidden_states = vec![
            arr2(&[[0.1], [0.2], [0.3]]),
            arr2(&[[0.0], [0.1], [0.2]]),
        ];

        let outputs = network.forward(&input, &hidden_states);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[3, 1]);
        assert_eq!(outputs[1].shape(), &[3, 1]);
    }

    #[test]
    fn test_gru_network_sequence() {
        let mut network = GRUNetwork::new(2, 3, 1);
        let sequence = vec![
            arr2(&[[1.0], [0.0]]),
            arr2(&[[0.0], [1.0]]),
            arr2(&[[-1.0], [0.5]]),
        ];

        let (outputs, caches) = network.forward_sequence_with_cache(&sequence);
        
        assert_eq!(outputs.len(), 3);
        assert_eq!(caches.len(), 3);
        
        for (output, _) in &outputs {
            assert_eq!(output.shape(), &[3, 1]);
        }
    }

    #[test]
    fn test_gru_network_with_dropout() {
        let mut network = GRUNetwork::new(2, 3, 2)
            .with_input_dropout(0.2, true)
            .with_recurrent_dropout(0.3, false)
            .with_output_dropout(0.1);

        let input = arr2(&[[1.0], [0.5]]);
        let hidden_states = vec![
            arr2(&[[0.1], [0.2], [0.3]]),
            arr2(&[[0.0], [0.1], [0.2]]),
        ];

        // Test training mode
        network.train();
        let outputs_train = network.forward(&input, &hidden_states);

        // Test evaluation mode
        network.eval();
        let outputs_eval = network.forward(&input, &hidden_states);

        assert_eq!(outputs_train.len(), 2);
        assert_eq!(outputs_eval.len(), 2);
    }

    #[test]
    fn test_gru_network_layer_dropout() {
        let layer_configs = vec![
            LayerDropoutConfig::new().with_input_dropout(0.1, false),
            LayerDropoutConfig::new().with_recurrent_dropout(0.2, true),
        ];

        let network = GRUNetwork::new(2, 3, 2)
            .with_layer_dropout(layer_configs);

        assert_eq!(network.cells.len(), 2);
    }
} 