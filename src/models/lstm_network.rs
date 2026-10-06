use crate::layers::lstm_cell::{masked, LSTMCell, LSTMCellCache, LSTMCellGradients};
use crate::optimizers::Optimizer;
use ndarray::Array2;

/// Hidden and cell states of every layer, each of shape (hidden_size, batch).
#[derive(Clone, Debug, PartialEq)]
pub struct LSTMState {
    pub h: Vec<Array2<f64>>,
    pub c: Vec<Array2<f64>>,
}

/// Values saved by one network time step.
#[derive(Clone)]
pub struct LSTMNetworkCache {
    pub cell_caches: Vec<LSTMCellCache>,
    /// Inverted-dropout masks applied between layer `i` and layer `i + 1`.
    pub output_dropout_masks: Vec<Option<Array2<f64>>>,
}

/// Multi-layer LSTM network for sequence modeling with dropout support
///
/// Stacks multiple LSTM cells where the output of layer i becomes
/// the input to layer i+1. Supports both inference and training with
/// configurable dropout regularization.
#[derive(Clone)]
pub struct LSTMNetwork {
    cells: Vec<LSTMCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub is_training: bool,
}

impl LSTMNetwork {
    /// Creates a new multi-layer LSTM network
    ///
    /// First layer accepts `input_size` dimensions, subsequent layers
    /// accept `hidden_size` dimensions from the previous layer.
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::new();

        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            cells.push(LSTMCell::new(layer_input_size, hidden_size));
        }

        LSTMNetwork {
            cells,
            input_size,
            hidden_size,
            num_layers,
            is_training: true,
        }
    }

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
        for (i, cell) in self.cells.iter_mut().enumerate() {
            if i < self.num_layers - 1 {
                *cell = cell.clone().with_output_dropout(dropout_rate);
            }
        }
        self
    }

    pub fn with_zoneout(mut self, cell_zoneout_rate: f64, hidden_zoneout_rate: f64) -> Self {
        for cell in &mut self.cells {
            *cell = cell
                .clone()
                .with_zoneout(cell_zoneout_rate, hidden_zoneout_rate);
        }
        self
    }

    /// Dropout on the candidate update of every layer (Semeniuta et al., 2016).
    pub fn with_cell_update_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        for cell in &mut self.cells {
            *cell = cell
                .clone()
                .with_cell_update_dropout(dropout_rate, variational);
        }
        self
    }

    pub fn with_layer_dropout(mut self, layer_configs: Vec<LayerDropoutConfig>) -> Self {
        for (i, config) in layer_configs.into_iter().enumerate() {
            if i < self.cells.len() {
                let mut cell = self.cells[i].clone();

                if let Some((rate, variational)) = config.input_dropout {
                    cell = cell.with_input_dropout(rate, variational);
                }
                if let Some((rate, variational)) = config.recurrent_dropout {
                    cell = cell.with_recurrent_dropout(rate, variational);
                }
                if let Some(rate) = config.output_dropout {
                    cell = cell.with_output_dropout(rate);
                }
                if let Some((cell_rate, hidden_rate)) = config.zoneout {
                    cell = cell.with_zoneout(cell_rate, hidden_rate);
                }
                if let Some((rate, variational)) = config.cell_update_dropout {
                    cell = cell.with_cell_update_dropout(rate, variational);
                }

                self.cells[i] = cell;
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

    /// Creates a network from existing cells (used for deserialization)
    pub fn from_cells(
        cells: Vec<LSTMCell>,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Self {
        LSTMNetwork {
            cells,
            input_size,
            hidden_size,
            num_layers,
            is_training: true,
        }
    }

    /// Get reference to the cells (used for serialization)
    pub fn get_cells(&self) -> &[LSTMCell] {
        &self.cells
    }

    /// Get mutable reference to the cells (for training mode changes)
    pub fn get_cells_mut(&mut self) -> &mut [LSTMCell] {
        &mut self.cells
    }

    /// All-zero state for `batch_size` sequences.
    pub fn zero_state(&self, batch_size: usize) -> LSTMState {
        let zeros = || {
            (0..self.num_layers)
                .map(|_| Array2::zeros((self.hidden_size, batch_size)))
                .collect()
        };
        LSTMState {
            h: zeros(),
            c: zeros(),
        }
    }

    /// Clears variational dropout masks so that the next sequence samples new ones.
    pub fn reset_dropout_masks(&mut self) {
        for cell in &mut self.cells {
            cell.reset_dropout_masks();
        }
    }

    /// One time step. `input` has shape (input_size, batch). Returns the output of
    /// the top layer and the new state of every layer.
    pub fn forward(&mut self, input: &Array2<f64>, state: &LSTMState) -> (Array2<f64>, LSTMState) {
        let (output, state, _) = self.forward_with_cache(input, state);
        (output, state)
    }

    /// One time step that also returns the values needed by `backward_sequence`.
    pub fn forward_with_cache(
        &mut self,
        input: &Array2<f64>,
        state: &LSTMState,
    ) -> (Array2<f64>, LSTMState, LSTMNetworkCache) {
        assert_eq!(
            state.h.len(),
            self.num_layers,
            "state must hold one entry per layer"
        );
        let mut layer_input = input.clone();
        let mut next = LSTMState {
            h: Vec::with_capacity(self.num_layers),
            c: Vec::with_capacity(self.num_layers),
        };
        let mut cell_caches = Vec::with_capacity(self.num_layers);
        let mut output_dropout_masks = Vec::with_capacity(self.num_layers);

        let last = self.num_layers - 1;
        for (l, cell) in self.cells.iter_mut().enumerate() {
            let (hy, cy, cache) = cell.forward_with_cache(&layer_input, &state.h[l], &state.c[l]);
            let mask = if l < last {
                cell.output_dropout_mask(hy.raw_dim())
            } else {
                None
            };
            layer_input = masked(&hy, &mask);
            next.h.push(hy);
            next.c.push(cy);
            cell_caches.push(cache);
            output_dropout_masks.push(mask);
        }

        let cache = LSTMNetworkCache {
            cell_caches,
            output_dropout_masks,
        };
        (layer_input, next, cache)
    }

    /// Runs a whole sequence from a zero state. Each element of `sequence` has shape
    /// (input_size, batch); returns the top-layer output at every step.
    pub fn forward_sequence(&mut self, sequence: &[Array2<f64>]) -> Vec<Array2<f64>> {
        self.forward_sequence_with_cache(sequence)
            .0
            .into_iter()
            .map(|(h, _)| h)
            .collect()
    }

    /// Runs a whole sequence from a zero state and keeps the caches for
    /// `backward_sequence`. Returns the top-layer hidden and cell state at every step.
    pub fn forward_sequence_with_cache(
        &mut self,
        sequence: &[Array2<f64>],
    ) -> (Vec<(Array2<f64>, Array2<f64>)>, Vec<LSTMNetworkCache>) {
        let batch_size = sequence.first().map_or(1, |x| x.ncols());
        let mut state = self.zero_state(batch_size);
        self.reset_dropout_masks();

        let mut outputs = Vec::with_capacity(sequence.len());
        let mut caches = Vec::with_capacity(sequence.len());
        for input in sequence {
            let (output, next, cache) = self.forward_with_cache(input, &state);
            outputs.push((output, next.c[self.num_layers - 1].clone()));
            caches.push(cache);
            state = next;
        }
        (outputs, caches)
    }

    /// Backpropagation through time over a sequence processed by
    /// `forward_sequence_with_cache`.
    ///
    /// `d_outputs[t]` is the gradient of the loss with respect to the top-layer
    /// output at step `t`. Returns the parameter gradients of every layer, summed
    /// over time and batch, and the gradient with respect to every input.
    pub fn backward_sequence(
        &self,
        d_outputs: &[Array2<f64>],
        caches: &[LSTMNetworkCache],
    ) -> (Vec<LSTMCellGradients>, Vec<Array2<f64>>) {
        assert_eq!(
            d_outputs.len(),
            caches.len(),
            "one output gradient per time step"
        );
        let mut gradients = self.zero_gradients();
        let mut d_inputs = vec![Array2::zeros((0, 0)); caches.len()];
        let batch_size = d_outputs.first().map_or(1, |d| d.ncols());
        let mut dh_next = self.zero_state(batch_size);

        let last = self.num_layers - 1;
        for t in (0..caches.len()).rev() {
            let mut d_above = d_outputs[t].clone();
            for l in (0..self.num_layers).rev() {
                let d_out = if l < last {
                    masked(&d_above, &caches[t].output_dropout_masks[l])
                } else {
                    d_above
                };
                let dh = &d_out + &dh_next.h[l];
                let (g, dx, dhx, dcx) =
                    self.cells[l].backward(&dh, &dh_next.c[l], &caches[t].cell_caches[l]);
                gradients[l].accumulate(&g);
                dh_next.h[l] = dhx;
                dh_next.c[l] = dcx;
                d_above = dx;
            }
            d_inputs[t] = d_above;
        }
        (gradients, d_inputs)
    }

    /// Update parameters for all layers using computed gradients
    pub fn update_parameters<O: Optimizer>(
        &mut self,
        gradients: &[LSTMCellGradients],
        optimizer: &mut O,
    ) {
        for (i, (cell, cell_gradients)) in self.cells.iter_mut().zip(gradients.iter()).enumerate() {
            let prefix = format!("layer_{}", i);
            cell.update_parameters(cell_gradients, optimizer, &prefix);
        }
    }

    /// Initialize zero gradients for all layers
    pub fn zero_gradients(&self) -> Vec<LSTMCellGradients> {
        self.cells
            .iter()
            .map(|cell| cell.zero_gradients())
            .collect()
    }

    /// Processes sequences of different lengths as one batch.
    ///
    /// Every element of every sequence has shape (input_size, 1). Shorter sequences
    /// are padded with zeros; their outputs after the last real step are discarded.
    pub fn forward_batch_sequences(
        &mut self,
        batch_sequences: &[Vec<Array2<f64>>],
    ) -> Vec<Vec<(Array2<f64>, Array2<f64>)>> {
        let padded = pad_batch(batch_sequences, self.input_size);
        let (outputs, _) = self.forward_sequence_with_cache(&padded);
        batch_sequences
            .iter()
            .enumerate()
            .map(|(b, seq)| {
                outputs
                    .iter()
                    .take(seq.len())
                    .map(|(h, c)| (column(h, b), column(c, b)))
                    .collect()
            })
            .collect()
    }
}

/// Stacks sequences of column vectors into zero-padded (features, batch) matrices.
pub fn pad_batch(batch_sequences: &[Vec<Array2<f64>>], features: usize) -> Vec<Array2<f64>> {
    let max_len = batch_sequences.iter().map(Vec::len).max().unwrap_or(0);
    (0..max_len)
        .map(|t| {
            let mut step = Array2::zeros((features, batch_sequences.len()));
            for (b, seq) in batch_sequences.iter().enumerate() {
                if let Some(x) = seq.get(t) {
                    step.column_mut(b).assign(&x.column(0));
                }
            }
            step
        })
        .collect()
}

fn column(m: &Array2<f64>, b: usize) -> Array2<f64> {
    m.column(b).to_owned().insert_axis(ndarray::Axis(1))
}

/// Configuration for layer-specific dropout settings
#[derive(Clone, Debug)]
pub struct LayerDropoutConfig {
    pub input_dropout: Option<(f64, bool)>, // (rate, variational)
    pub recurrent_dropout: Option<(f64, bool)>, // (rate, variational)
    pub output_dropout: Option<f64>,        // rate
    pub zoneout: Option<(f64, f64)>,        // (cell_rate, hidden_rate)
    pub cell_update_dropout: Option<(f64, bool)>, // (rate, variational)
}

impl Default for LayerDropoutConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerDropoutConfig {
    pub fn new() -> Self {
        LayerDropoutConfig {
            input_dropout: None,
            recurrent_dropout: None,
            output_dropout: None,
            zoneout: None,
            cell_update_dropout: None,
        }
    }

    pub fn with_input_dropout(mut self, rate: f64, variational: bool) -> Self {
        self.input_dropout = Some((rate, variational));
        self
    }

    pub fn with_recurrent_dropout(mut self, rate: f64, variational: bool) -> Self {
        self.recurrent_dropout = Some((rate, variational));
        self
    }

    pub fn with_output_dropout(mut self, rate: f64) -> Self {
        self.output_dropout = Some(rate);
        self
    }

    pub fn with_zoneout(mut self, cell_rate: f64, hidden_rate: f64) -> Self {
        self.zoneout = Some((cell_rate, hidden_rate));
        self
    }

    pub fn with_cell_update_dropout(mut self, rate: f64, variational: bool) -> Self {
        self.cell_update_dropout = Some((rate, variational));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_lstm_network_forward() {
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 2;
        let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let state = network.zero_state(1);

        let (hy, next) = network.forward(&input, &state);
        let cy = next.c[num_layers - 1].clone();

        assert_eq!(hy.shape(), &[hidden_size, 1]);
        assert_eq!(cy.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_lstm_network_with_dropout() {
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 2;
        let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
            .with_input_dropout(0.2, true) // Variational input dropout
            .with_recurrent_dropout(0.3, true) // Variational recurrent dropout
            .with_output_dropout(0.1)
            .with_zoneout(0.1, 0.1);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let state = network.zero_state(1);

        // Test training mode
        network.train();
        let (hy_train, next) = network.forward(&input, &state);
        let cy_train = next.c[num_layers - 1].clone();

        // Test evaluation mode
        network.eval();
        let (hy_eval, next) = network.forward(&input, &state);
        let cy_eval = next.c[num_layers - 1].clone();

        assert_eq!(hy_train.shape(), &[hidden_size, 1]);
        assert_eq!(cy_train.shape(), &[hidden_size, 1]);
        assert_eq!(hy_eval.shape(), &[hidden_size, 1]);
        assert_eq!(cy_eval.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_layer_specific_dropout() {
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 2;

        let layer_configs = vec![
            LayerDropoutConfig::new()
                .with_input_dropout(0.2, true)
                .with_recurrent_dropout(0.3, true),
            LayerDropoutConfig::new()
                .with_output_dropout(0.1)
                .with_zoneout(0.1, 0.1),
        ];

        let mut network =
            LSTMNetwork::new(input_size, hidden_size, num_layers).with_layer_dropout(layer_configs);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let state = network.zero_state(1);

        let (hy, next) = network.forward(&input, &state);
        let cy = next.c[num_layers - 1].clone();

        assert_eq!(hy.shape(), &[hidden_size, 1]);
        assert_eq!(cy.shape(), &[hidden_size, 1]);
    }
}
