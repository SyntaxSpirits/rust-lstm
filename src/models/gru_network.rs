use crate::layers::gru_cell::{GRUCell, GRUCellCache, GRUCellGradients};
use crate::layers::lstm_cell::masked;
use crate::optimizers::Optimizer;
use ndarray::Array2;

/// Cache for GRU network forward pass
#[derive(Clone)]
pub struct GRUNetworkCache {
    pub caches: Vec<GRUCellCache>,
    /// Inverted-dropout masks applied between layer `i` and layer `i + 1`.
    pub output_dropout_masks: Vec<Option<Array2<f64>>>,
}

/// Configuration for layer-specific dropout settings
#[derive(Clone)]
pub struct LayerDropoutConfig {
    pub input_dropout_rate: f64,
    pub input_variational: bool,
    pub recurrent_dropout_rate: f64,
    pub recurrent_variational: bool,
    pub output_dropout_rate: f64,
    pub candidate_dropout_rate: f64,
    pub candidate_variational: bool,
    pub zoneout_rate: f64,
}

impl Default for LayerDropoutConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerDropoutConfig {
    pub fn new() -> Self {
        LayerDropoutConfig {
            input_dropout_rate: 0.0,
            input_variational: false,
            recurrent_dropout_rate: 0.0,
            recurrent_variational: false,
            output_dropout_rate: 0.0,
            candidate_dropout_rate: 0.0,
            candidate_variational: false,
            zoneout_rate: 0.0,
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

    pub fn with_candidate_dropout(mut self, rate: f64, variational: bool) -> Self {
        self.candidate_dropout_rate = rate;
        self.candidate_variational = variational;
        self
    }

    pub fn with_zoneout(mut self, rate: f64) -> Self {
        self.zoneout_rate = rate;
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
            *cell = cell
                .clone()
                .with_recurrent_dropout(dropout_rate, variational);
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

    /// Dropout on the candidate state of every layer (Semeniuta et al., 2016).
    pub fn with_candidate_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in &mut self.cells {
            *cell = cell
                .clone()
                .with_candidate_dropout(dropout_rate, variational);
        }
        self
    }

    /// Zoneout of the hidden state of every layer (Krueger et al., 2017).
    pub fn with_zoneout(mut self, rate: f64) -> Self {
        for cell in &mut self.cells {
            *cell = cell.clone().with_zoneout(rate);
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
                self.cells[i] = self.cells[i]
                    .clone()
                    .with_input_dropout(config.input_dropout_rate, config.input_variational);
            }
            if config.recurrent_dropout_rate > 0.0 {
                self.cells[i] = self.cells[i].clone().with_recurrent_dropout(
                    config.recurrent_dropout_rate,
                    config.recurrent_variational,
                );
            }
            if config.zoneout_rate > 0.0 {
                self.cells[i] = self.cells[i].clone().with_zoneout(config.zoneout_rate);
            }
            if config.candidate_dropout_rate > 0.0 {
                self.cells[i] = self.cells[i].clone().with_candidate_dropout(
                    config.candidate_dropout_rate,
                    config.candidate_variational,
                );
            }
            if config.output_dropout_rate > 0.0 && i < self.num_layers - 1 {
                self.cells[i] = self.cells[i]
                    .clone()
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

    /// Clears variational dropout masks so that the next sequence samples new ones.
    pub fn reset_dropout_masks(&mut self) {
        for cell in &mut self.cells {
            cell.reset_dropout_masks();
        }
    }

    /// Forward pass for a single time step; returns the new hidden state of every layer.
    pub fn forward(&mut self, input: &Array2<f64>, hx: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.forward_with_cache(input, hx).0
    }

    /// One time step that also returns the values needed by `backward_sequence`.
    pub fn forward_with_cache(
        &mut self,
        input: &Array2<f64>,
        hx: &[Array2<f64>],
    ) -> (Vec<Array2<f64>>, GRUNetworkCache) {
        if hx.len() != self.num_layers {
            panic!("Number of hidden states must match number of layers");
        }

        let last = self.num_layers - 1;
        let mut layer_input = input.clone();
        let mut states = Vec::with_capacity(self.num_layers);
        let mut caches = Vec::with_capacity(self.num_layers);
        let mut output_dropout_masks = Vec::with_capacity(self.num_layers);
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let (hy, cache) = cell.forward_with_cache(&layer_input, &hx[i]);
            let mask = if i < last {
                cell.output_dropout_mask(hy.raw_dim())
            } else {
                None
            };
            layer_input = masked(&hy, &mask);
            states.push(hy);
            caches.push(cache);
            output_dropout_masks.push(mask);
        }
        let cache = GRUNetworkCache {
            caches,
            output_dropout_masks,
        };
        (states, cache)
    }

    /// Forward pass for a sequence with caching for training.
    ///
    /// Returns, for every step, the top-layer output and the states of all layers.
    pub fn forward_sequence_with_cache(
        &mut self,
        sequence: &[Array2<f64>],
    ) -> (Vec<(Array2<f64>, Vec<Array2<f64>>)>, Vec<GRUNetworkCache>) {
        let batch_size = sequence.first().map_or(1, |x| x.ncols());
        let mut hidden_states: Vec<Array2<f64>> = (0..self.num_layers)
            .map(|_| Array2::zeros((self.hidden_size, batch_size)))
            .collect();
        self.reset_dropout_masks();

        let mut all_outputs = Vec::with_capacity(sequence.len());
        let mut all_caches = Vec::with_capacity(sequence.len());
        for input in sequence {
            let (states, cache) = self.forward_with_cache(input, &hidden_states);
            all_outputs.push((states[self.num_layers - 1].clone(), states.clone()));
            all_caches.push(cache);
            hidden_states = states;
        }
        (all_outputs, all_caches)
    }

    /// Backpropagation through time over a sequence processed by
    /// `forward_sequence_with_cache`.
    ///
    /// `d_outputs[t]` is the gradient of the loss with respect to the top-layer
    /// output at step `t`. Returns the parameter gradients of every layer and the
    /// gradient with respect to every input.
    pub fn backward_sequence(
        &self,
        d_outputs: &[Array2<f64>],
        caches: &[GRUNetworkCache],
    ) -> (Vec<GRUCellGradients>, Vec<Array2<f64>>) {
        assert_eq!(
            d_outputs.len(),
            caches.len(),
            "one output gradient per time step"
        );
        let mut gradients = self.zero_gradients();
        let mut d_inputs = vec![Array2::zeros((0, 0)); caches.len()];
        let batch_size = d_outputs.first().map_or(1, |d| d.ncols());
        let mut dh_next: Vec<Array2<f64>> = (0..self.num_layers)
            .map(|_| Array2::zeros((self.hidden_size, batch_size)))
            .collect();

        let last = self.num_layers - 1;
        for t in (0..caches.len()).rev() {
            let mut d_above = d_outputs[t].clone();
            for l in (0..self.num_layers).rev() {
                let d_out = if l < last {
                    masked(&d_above, &caches[t].output_dropout_masks[l])
                } else {
                    d_above
                };
                let dh = &d_out + &dh_next[l];
                let (g, dx, dhx) = self.cells[l].backward(&dh, &caches[t].caches[l]);
                gradients[l].accumulate(&g);
                dh_next[l] = dhx;
                d_above = dx;
            }
            d_inputs[t] = d_above;
        }
        (gradients, d_inputs)
    }

    /// Update parameters using optimizer
    pub fn update_parameters<O: Optimizer>(
        &mut self,
        gradients: &[GRUCellGradients],
        optimizer: &mut O,
    ) {
        for (i, (cell, grad)) in self.cells.iter_mut().zip(gradients.iter()).enumerate() {
            cell.update_parameters(grad, optimizer, &format!("layer_{}", i));
        }
    }

    /// Initialize zero gradients for all layers
    pub fn zero_gradients(&self) -> Vec<GRUCellGradients> {
        self.cells
            .iter()
            .map(|cell| cell.zero_gradients())
            .collect()
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
        let hidden_states = vec![arr2(&[[0.1], [0.2], [0.3]]), arr2(&[[0.0], [0.1], [0.2]])];

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
        let hidden_states = vec![arr2(&[[0.1], [0.2], [0.3]]), arr2(&[[0.0], [0.1], [0.2]])];

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

        let network = GRUNetwork::new(2, 3, 2).with_layer_dropout(layer_configs);

        assert_eq!(network.cells.len(), 2);
    }
}
