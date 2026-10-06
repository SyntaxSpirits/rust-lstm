use crate::layers::lstm_cell::{masked, LSTMCell, LSTMCellCache, LSTMCellGradients};
use crate::optimizers::Optimizer;
use ndarray::Array2;

/// Cache for bidirectional LSTM forward pass
///
/// All vectors are indexed `[layer][time]` in the original time order.
#[derive(Clone)]
pub struct BiLSTMNetworkCache {
    pub forward_caches: Vec<Vec<LSTMCellCache>>,
    pub backward_caches: Vec<Vec<LSTMCellCache>>,
    pub forward_output_masks: Vec<Vec<Option<Array2<f64>>>>,
    pub backward_output_masks: Vec<Vec<Option<Array2<f64>>>>,
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
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        combine_mode: CombineMode,
    ) -> Self {
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
            *cell = cell
                .clone()
                .with_recurrent_dropout(dropout_rate, variational);
        }
        for cell in &mut self.backward_cells {
            *cell = cell
                .clone()
                .with_recurrent_dropout(dropout_rate, variational);
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

    pub fn with_cell_update_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in self
            .forward_cells
            .iter_mut()
            .chain(self.backward_cells.iter_mut())
        {
            *cell = cell
                .clone()
                .with_cell_update_dropout(dropout_rate, variational);
        }
        self
    }

    pub fn with_zoneout(mut self, cell_zoneout_rate: f64, hidden_zoneout_rate: f64) -> Self {
        for cell in &mut self.forward_cells {
            *cell = cell
                .clone()
                .with_zoneout(cell_zoneout_rate, hidden_zoneout_rate);
        }
        for cell in &mut self.backward_cells {
            *cell = cell
                .clone()
                .with_zoneout(cell_zoneout_rate, hidden_zoneout_rate);
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
                let mut combined =
                    Array2::zeros((forward.nrows() + backward.nrows(), forward.ncols()));
                combined
                    .slice_mut(ndarray::s![..forward.nrows(), ..])
                    .assign(forward);
                combined
                    .slice_mut(ndarray::s![forward.nrows().., ..])
                    .assign(backward);
                combined
            }
            CombineMode::Sum => forward + backward,
            CombineMode::Average => (forward + backward) * 0.5,
        }
    }

    /// Forward pass for a complete sequence
    ///
    /// Each layer runs one LSTM from start to end and another from end to start;
    /// their outputs are combined and fed to the next layer.
    pub fn forward_sequence(&mut self, sequence: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.forward_sequence_with_cache(sequence).0
    }

    /// Forward pass with caching for training
    pub fn forward_sequence_with_cache(
        &mut self,
        sequence: &[Array2<f64>],
    ) -> (Vec<Array2<f64>>, BiLSTMNetworkCache) {
        let seq_len = sequence.len();
        let batch_size = sequence.first().map_or(1, |x| x.ncols());
        let mut cache = BiLSTMNetworkCache {
            forward_caches: Vec::with_capacity(self.num_layers),
            backward_caches: Vec::with_capacity(self.num_layers),
            forward_output_masks: Vec::with_capacity(self.num_layers),
            backward_output_masks: Vec::with_capacity(self.num_layers),
        };
        if seq_len == 0 {
            return (Vec::new(), cache);
        }

        let last = self.num_layers - 1;
        let mut layer_input = sequence.to_vec();
        for l in 0..self.num_layers {
            let order: Vec<usize> = (0..seq_len).collect();
            let reversed: Vec<usize> = (0..seq_len).rev().collect();
            let (f_out, f_cache, f_masks) = run_direction(
                &mut self.forward_cells[l],
                &layer_input,
                &order,
                batch_size,
                l < last,
            );
            let (b_out, b_cache, b_masks) = run_direction(
                &mut self.backward_cells[l],
                &layer_input,
                &reversed,
                batch_size,
                l < last,
            );
            layer_input = f_out
                .iter()
                .zip(&b_out)
                .map(|(f, b)| self.combine_outputs(f, b))
                .collect();
            cache.forward_caches.push(f_cache);
            cache.backward_caches.push(b_cache);
            cache.forward_output_masks.push(f_masks);
            cache.backward_output_masks.push(b_masks);
        }
        (layer_input, cache)
    }

    /// Backpropagation through time for both directions.
    ///
    /// `d_outputs[t]` is the gradient of the loss with respect to the combined output
    /// at step `t`. Returns the gradients of the forward cells, of the backward cells
    /// and of every input.
    pub fn backward_sequence(
        &self,
        d_outputs: &[Array2<f64>],
        cache: &BiLSTMNetworkCache,
    ) -> (
        Vec<LSTMCellGradients>,
        Vec<LSTMCellGradients>,
        Vec<Array2<f64>>,
    ) {
        let (mut f_grads, mut b_grads) = self.zero_gradients();
        let seq_len = d_outputs.len();
        let mut d_layer = d_outputs.to_vec();
        let h = self.hidden_size;

        for l in (0..self.num_layers).rev() {
            let (d_f, d_b): (Vec<_>, Vec<_>) = d_layer
                .iter()
                .enumerate()
                .map(|(t, d)| {
                    let (df, db) = match self.combine_mode {
                        CombineMode::Concat => (
                            d.slice(ndarray::s![..h, ..]).to_owned(),
                            d.slice(ndarray::s![h.., ..]).to_owned(),
                        ),
                        CombineMode::Sum => (d.clone(), d.clone()),
                        CombineMode::Average => (d * 0.5, d * 0.5),
                    };
                    (
                        masked(&df, &cache.forward_output_masks[l][t]),
                        masked(&db, &cache.backward_output_masks[l][t]),
                    )
                })
                .unzip();

            let forward_bptt: Vec<usize> = (0..seq_len).rev().collect();
            let backward_bptt: Vec<usize> = (0..seq_len).collect();
            let dx_f = backprop_direction(
                &self.forward_cells[l],
                &d_f,
                &cache.forward_caches[l],
                &forward_bptt,
                &mut f_grads[l],
            );
            let dx_b = backprop_direction(
                &self.backward_cells[l],
                &d_b,
                &cache.backward_caches[l],
                &backward_bptt,
                &mut b_grads[l],
            );
            d_layer = dx_f.iter().zip(&dx_b).map(|(a, b)| a + b).collect();
        }
        (f_grads, b_grads, d_layer)
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
    pub fn update_parameters<O: Optimizer>(
        &mut self,
        forward_gradients: &[LSTMCellGradients],
        backward_gradients: &[LSTMCellGradients],
        optimizer: &mut O,
    ) {
        // Update forward cells
        for (i, (cell, gradients)) in self
            .forward_cells
            .iter_mut()
            .zip(forward_gradients.iter())
            .enumerate()
        {
            cell.update_parameters(gradients, optimizer, &format!("forward_layer_{}", i));
        }

        // Update backward cells
        for (i, (cell, gradients)) in self
            .backward_cells
            .iter_mut()
            .zip(backward_gradients.iter())
            .enumerate()
        {
            cell.update_parameters(gradients, optimizer, &format!("backward_layer_{}", i));
        }
    }

    /// Zero gradients for all cells
    pub fn zero_gradients(&self) -> (Vec<LSTMCellGradients>, Vec<LSTMCellGradients>) {
        let forward_gradients: Vec<_> = self
            .forward_cells
            .iter()
            .map(|cell| cell.zero_gradients())
            .collect();

        let backward_gradients: Vec<_> = self
            .backward_cells
            .iter()
            .map(|cell| cell.zero_gradients())
            .collect();

        (forward_gradients, backward_gradients)
    }
}

/// Runs one direction of a layer over `inputs` in the given time order and returns
/// outputs, caches and inter-layer dropout masks, all in the original time order.
fn run_direction(
    cell: &mut LSTMCell,
    inputs: &[Array2<f64>],
    order: &[usize],
    batch_size: usize,
    drop_output: bool,
) -> (
    Vec<Array2<f64>>,
    Vec<LSTMCellCache>,
    Vec<Option<Array2<f64>>>,
) {
    cell.reset_dropout_masks();
    let n = inputs.len();
    let mut outputs = vec![Array2::zeros((0, 0)); n];
    let mut caches: Vec<Option<LSTMCellCache>> = vec![None; n];
    let mut masks = vec![None; n];
    let mut hx = Array2::zeros((cell.hidden_size, batch_size));
    let mut cx = Array2::zeros((cell.hidden_size, batch_size));
    for &t in order {
        let (hy, cy, cache) = cell.forward_with_cache(&inputs[t], &hx, &cx);
        let mask = if drop_output {
            cell.output_dropout_mask(hy.raw_dim())
        } else {
            None
        };
        outputs[t] = masked(&hy, &mask);
        masks[t] = mask;
        caches[t] = Some(cache);
        hx = hy;
        cx = cy;
    }
    (outputs, caches.into_iter().flatten().collect(), masks)
}

/// BPTT for one direction; `order` visits the steps in reverse processing order.
fn backprop_direction(
    cell: &LSTMCell,
    d_outputs: &[Array2<f64>],
    caches: &[LSTMCellCache],
    order: &[usize],
    gradients: &mut LSTMCellGradients,
) -> Vec<Array2<f64>> {
    let batch_size = d_outputs.first().map_or(1, |d| d.ncols());
    let mut dh_next = Array2::zeros((cell.hidden_size, batch_size));
    let mut dc_next = Array2::zeros((cell.hidden_size, batch_size));
    let mut d_inputs = vec![Array2::zeros((0, 0)); d_outputs.len()];
    for &t in order {
        let dh = &d_outputs[t] + &dh_next;
        let (g, dx, dhx, dcx) = cell.backward(&dh, &dc_next, &caches[t]);
        gradients.accumulate(&g);
        dh_next = dhx;
        dc_next = dcx;
        d_inputs[t] = dx;
    }
    d_inputs
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
