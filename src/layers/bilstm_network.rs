use ndarray::Array2;
use crate::layers::lstm_cell::{LSTMCell, LSTMCellGradients, LSTMCellCache};
use crate::optimizers::Optimizer;

/// Cache for bidirectional LSTM forward pass
#[derive(Clone)]
pub struct BiLSTMNetworkCache {
    pub forward_caches: Vec<LSTMCellCache>,
    pub backward_caches: Vec<LSTMCellCache>,
}

/// Configuration for combining forward and backward outputs
#[derive(Clone, Debug)]
pub enum CombineMode {
    Concat,
    Sum,
    Average,
}

/// Bidirectional LSTM network for sequence modeling
#[derive(Clone)]
pub struct BiLSTMNetwork {
    forward_cells: Vec<LSTMCell>,
    backward_cells: Vec<LSTMCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub combine_mode: CombineMode,
    pub is_training: bool,
}

impl BiLSTMNetwork {
    /// Creates a new bidirectional LSTM network
    /// 
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state for each direction
    /// * `num_layers` - Number of bidirectional layers
    /// * `combine_mode` - How to combine forward and backward outputs
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, combine_mode: CombineMode) -> Self {
        let mut forward_cells = Vec::new();
        let mut backward_cells = Vec::new();

        for i in 0..num_layers {
            let layer_input_size = if i == 0 {
                input_size
            } else {
                match combine_mode {
                    CombineMode::Concat => 2 * hidden_size,
                    CombineMode::Sum | CombineMode::Average => hidden_size,
                }
            };

            forward_cells.push(LSTMCell::new(layer_input_size, hidden_size));
            backward_cells.push(LSTMCell::new(layer_input_size, hidden_size));
        }
        
        BiLSTMNetwork { 
            forward_cells,
            backward_cells,
            input_size,
            hidden_size,
            num_layers,
            combine_mode,
            is_training: true,
        }
    }

    /// Create BiLSTM with concatenated outputs (most common)
    pub fn new_concat(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::new(input_size, hidden_size, num_layers, CombineMode::Concat)
    }

    /// Create BiLSTM with summed outputs
    pub fn new_sum(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::new(input_size, hidden_size, num_layers, CombineMode::Sum)
    }

    /// Create BiLSTM with averaged outputs
    pub fn new_average(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::new(input_size, hidden_size, num_layers, CombineMode::Average)
    }

    /// Get the output size based on combine mode
    pub fn output_size(&self) -> usize {
        match self.combine_mode {
            CombineMode::Concat => 2 * self.hidden_size,
            CombineMode::Sum | CombineMode::Average => self.hidden_size,
        }
    }

    /// Apply dropout configuration to all cells
    pub fn with_input_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in &mut self.forward_cells {
            *cell = cell.clone().with_input_dropout(dropout_rate, variational);
        }
        for cell in &mut self.backward_cells {
            *cell = cell.clone().with_input_dropout(dropout_rate, variational);
        }
        self
    }

    pub fn with_recurrent_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in &mut self.forward_cells {
            *cell = cell.clone().with_recurrent_dropout(dropout_rate, variational);
        }
        for cell in &mut self.backward_cells {
            *cell = cell.clone().with_recurrent_dropout(dropout_rate, variational);
        }
        self
    }

    pub fn with_output_dropout(mut self, dropout_rate: f64) -> Self {
        // Apply output dropout to all layers except the last
        for (i, cell) in self.forward_cells.iter_mut().enumerate() {
            if i < self.num_layers - 1 {
                *cell = cell.clone().with_output_dropout(dropout_rate);
            }
        }
        for (i, cell) in self.backward_cells.iter_mut().enumerate() {
            if i < self.num_layers - 1 {
                *cell = cell.clone().with_output_dropout(dropout_rate);
            }
        }
        self
    }

    pub fn with_zoneout(mut self, cell_zoneout_rate: f64, hidden_zoneout_rate: f64) -> Self {
        for cell in &mut self.forward_cells {
            *cell = cell.clone().with_zoneout(cell_zoneout_rate, hidden_zoneout_rate);
        }
        for cell in &mut self.backward_cells {
            *cell = cell.clone().with_zoneout(cell_zoneout_rate, hidden_zoneout_rate);
        }
        self
    }

    pub fn train(&mut self) {
        self.is_training = true;
        for cell in &mut self.forward_cells {
            cell.train();
        }
        for cell in &mut self.backward_cells {
            cell.train();
        }
    }

    pub fn eval(&mut self) {
        self.is_training = false;
        for cell in &mut self.forward_cells {
            cell.eval();
        }
        for cell in &mut self.backward_cells {
            cell.eval();
        }
    }

    /// Combine forward and backward outputs according to the combine mode
    fn combine_outputs(&self, forward: &Array2<f64>, backward: &Array2<f64>) -> Array2<f64> {
        match self.combine_mode {
            CombineMode::Concat => {
                // Stack forward and backward outputs vertically
                let mut combined = Array2::zeros((forward.nrows() + backward.nrows(), forward.ncols()));
                combined.slice_mut(ndarray::s![..forward.nrows(), ..]).assign(forward);
                combined.slice_mut(ndarray::s![forward.nrows().., ..]).assign(backward);
                combined
            },
            CombineMode::Sum => forward + backward,
            CombineMode::Average => (forward + backward) * 0.5,
        }
    }

    /// Forward pass for a complete sequence
    /// 
    /// This is the main method for BiLSTM processing. It runs the forward direction
    /// from start to end, backward direction from end to start, then combines outputs.
    pub fn forward_sequence(&mut self, sequence: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let seq_len = sequence.len();
        if seq_len == 0 {
            return Vec::new();
        }

        // Process each layer sequentially
        let mut layer_input_sequence = sequence.to_vec();

        for layer_idx in 0..self.num_layers {
            let mut forward_outputs = Vec::new();
            let mut backward_outputs = Vec::new();

            // Initialize states for this layer
            let mut forward_hidden_state = Array2::zeros((self.hidden_size, 1));
            let mut forward_cell_state = Array2::zeros((self.hidden_size, 1));
            let mut backward_hidden_state = Array2::zeros((self.hidden_size, 1));
            let mut backward_cell_state = Array2::zeros((self.hidden_size, 1));

            // Forward direction
            for t in 0..seq_len {
                let (hy, cy) = self.forward_cells[layer_idx].forward(
                    &layer_input_sequence[t],
                    &forward_hidden_state,
                    &forward_cell_state
                );

                forward_hidden_state = hy.clone();
                forward_cell_state = cy;
                forward_outputs.push(hy);
            }

            // Backward direction
            for t in (0..seq_len).rev() {
                let (hy, cy) = self.backward_cells[layer_idx].forward(
                    &layer_input_sequence[t],
                    &backward_hidden_state,
                    &backward_cell_state
                );

                backward_hidden_state = hy.clone();
                backward_cell_state = cy;
                backward_outputs.push(hy);
            }

            // Reverse backward outputs to match forward sequence order
            backward_outputs.reverse();

            // Combine forward and backward outputs for this layer
            let mut combined_outputs = Vec::new();
            for (forward_out, backward_out) in forward_outputs.iter().zip(backward_outputs.iter()) {
                combined_outputs.push(self.combine_outputs(forward_out, backward_out));
            }

            // Output of this layer becomes input to next layer
            layer_input_sequence = combined_outputs;
        }

        layer_input_sequence
    }

    /// Forward pass with caching for training
    pub fn forward_sequence_with_cache(&mut self, sequence: &[Array2<f64>]) -> (Vec<Array2<f64>>, BiLSTMNetworkCache) {
        let seq_len = sequence.len();
        if seq_len == 0 {
            return (Vec::new(), BiLSTMNetworkCache {
                forward_caches: Vec::new(),
                backward_caches: Vec::new(),
            });
        }

        let mut all_forward_caches = Vec::new();
        let mut all_backward_caches = Vec::new();

        // Process each layer sequentially
        let mut layer_input_sequence = sequence.to_vec();

        for layer_idx in 0..self.num_layers {
            let mut forward_outputs = Vec::new();
            let mut backward_outputs = Vec::new();
            let mut forward_caches = Vec::new();
            let mut backward_caches = Vec::new();

            // Initialize states for this layer
            let mut forward_hidden_state = Array2::zeros((self.hidden_size, 1));
            let mut forward_cell_state = Array2::zeros((self.hidden_size, 1));
            let mut backward_hidden_state = Array2::zeros((self.hidden_size, 1));
            let mut backward_cell_state = Array2::zeros((self.hidden_size, 1));

            // Forward direction with caching
            for t in 0..seq_len {
                let (hy, cy, cache) = self.forward_cells[layer_idx].forward_with_cache(
                    &layer_input_sequence[t],
                    &forward_hidden_state,
                    &forward_cell_state
                );

                forward_hidden_state = hy.clone();
                forward_cell_state = cy;
                forward_outputs.push(hy);
                forward_caches.push(cache);
            }

            // Backward direction with caching
            for t in (0..seq_len).rev() {
                let (hy, cy, cache) = self.backward_cells[layer_idx].forward_with_cache(
                    &layer_input_sequence[t],
                    &backward_hidden_state,
                    &backward_cell_state
                );

                backward_hidden_state = hy.clone();
                backward_cell_state = cy;
                backward_outputs.push(hy);
                backward_caches.push(cache);
            }

            // Reverse backward outputs and caches
            backward_outputs.reverse();
            backward_caches.reverse();

            // Combine outputs for this layer
            let mut combined_outputs = Vec::new();
            for (forward_out, backward_out) in forward_outputs.iter().zip(backward_outputs.iter()) {
                combined_outputs.push(self.combine_outputs(forward_out, backward_out));
            }

            // Store caches for this layer
            all_forward_caches.extend(forward_caches);
            all_backward_caches.extend(backward_caches);

            // Output of this layer becomes input to next layer
            layer_input_sequence = combined_outputs;
        }

        let cache = BiLSTMNetworkCache {
            forward_caches: all_forward_caches,
            backward_caches: all_backward_caches,
        };

        (layer_input_sequence, cache)
    }

    /// Get references to forward and backward cells for serialization
    pub fn get_forward_cells(&self) -> &[LSTMCell] {
        &self.forward_cells
    }

    pub fn get_backward_cells(&self) -> &[LSTMCell] {
        &self.backward_cells
    }

    /// Get mutable references for training mode changes
    pub fn get_forward_cells_mut(&mut self) -> &mut [LSTMCell] {
        &mut self.forward_cells
    }

    pub fn get_backward_cells_mut(&mut self) -> &mut [LSTMCell] {
        &mut self.backward_cells
    }

    /// Update parameters for both directions
    pub fn update_parameters<O: Optimizer>(&mut self, 
                                         forward_gradients: &[LSTMCellGradients], 
                                         backward_gradients: &[LSTMCellGradients], 
                                         optimizer: &mut O) {
        // Update forward cells
        for (i, (cell, gradients)) in self.forward_cells.iter_mut().zip(forward_gradients.iter()).enumerate() {
            cell.update_parameters(gradients, optimizer, &format!("forward_layer_{}", i));
        }

        // Update backward cells
        for (i, (cell, gradients)) in self.backward_cells.iter_mut().zip(backward_gradients.iter()).enumerate() {
            cell.update_parameters(gradients, optimizer, &format!("backward_layer_{}", i));
        }
    }

    /// Zero gradients for all cells
    pub fn zero_gradients(&self) -> (Vec<LSTMCellGradients>, Vec<LSTMCellGradients>) {
        let forward_gradients: Vec<_> = self.forward_cells.iter()
            .map(|cell| cell.zero_gradients())
            .collect();

        let backward_gradients: Vec<_> = self.backward_cells.iter()
            .map(|cell| cell.zero_gradients())
            .collect();

        (forward_gradients, backward_gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_bilstm_creation() {
        let network = BiLSTMNetwork::new_concat(3, 5, 2);
        assert_eq!(network.input_size, 3);
        assert_eq!(network.hidden_size, 5);
        assert_eq!(network.num_layers, 2);
        assert_eq!(network.output_size(), 10); // 2 * hidden_size for concat mode
    }

    #[test]
    fn test_bilstm_combine_modes() {
        let forward = arr2(&[[1.0], [2.0]]);
        let backward = arr2(&[[3.0], [4.0]]);

        let concat_network = BiLSTMNetwork::new_concat(2, 2, 1);
        let concat_result = concat_network.combine_outputs(&forward, &backward);
        assert_eq!(concat_result.shape(), &[4, 1]);
        assert_eq!(concat_result[[0, 0]], 1.0);
        assert_eq!(concat_result[[1, 0]], 2.0);
        assert_eq!(concat_result[[2, 0]], 3.0);
        assert_eq!(concat_result[[3, 0]], 4.0);

        let sum_network = BiLSTMNetwork::new_sum(2, 2, 1);
        let sum_result = sum_network.combine_outputs(&forward, &backward);
        assert_eq!(sum_result.shape(), &[2, 1]);
        assert_eq!(sum_result[[0, 0]], 4.0);
        assert_eq!(sum_result[[1, 0]], 6.0);

        let avg_network = BiLSTMNetwork::new_average(2, 2, 1);
        let avg_result = avg_network.combine_outputs(&forward, &backward);
        assert_eq!(avg_result.shape(), &[2, 1]);
        assert_eq!(avg_result[[0, 0]], 2.0);
        assert_eq!(avg_result[[1, 0]], 3.0);
    }

    #[test]
    fn test_bilstm_forward_sequence() {
        let mut network = BiLSTMNetwork::new_concat(2, 3, 1);
        
        let sequence = vec![
            arr2(&[[1.0], [0.5]]),
            arr2(&[[0.8], [0.2]]),
            arr2(&[[0.3], [0.9]]),
        ];

        let outputs = network.forward_sequence(&sequence);
        
        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.shape(), &[6, 1]); // 2 * hidden_size for concat
        }
    }

    #[test]
    fn test_bilstm_training_mode() {
        let mut network = BiLSTMNetwork::new_concat(2, 3, 1)
            .with_input_dropout(0.1, false)
            .with_recurrent_dropout(0.2, true);

        // Test mode switching
        network.train();
        assert!(network.is_training);

        network.eval();
        assert!(!network.is_training);
    }
} 