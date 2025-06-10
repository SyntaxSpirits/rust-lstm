use ndarray::Array2;
use crate::layers::lstm_cell::{LSTMCell, LSTMCellGradients, LSTMCellCache};
use crate::optimizers::Optimizer;

/// Holds cached values for all layers during network forward pass
pub struct LSTMNetworkCache {
    pub cell_caches: Vec<LSTMCellCache>,
}

/// Multi-layer LSTM network for sequence modeling
/// 
/// Stacks multiple LSTM cells where the output of layer i becomes 
/// the input to layer i+1. Supports both inference and training.
pub struct LSTMNetwork {
    cells: Vec<LSTMCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
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
        }
    }

    /// Forward pass for inference (no caching)
    pub fn forward(&self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (hy, cy, _) = self.forward_with_cache(input, hx, cx);
        (hy, cy)
    }

    /// Forward pass with caching for training
    pub fn forward_with_cache(&self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>, LSTMNetworkCache) {
        let mut current_input = input.clone();
        let mut current_hx = hx.clone();
        let mut current_cx = cx.clone();
        let mut cell_caches = Vec::new();

        // Forward through each layer sequentially
        for cell in &self.cells {
            let (new_hx, new_cx, cache) = cell.forward_with_cache(&current_input, &current_hx, &current_cx);
            cell_caches.push(cache);

            // Layer i+1 input is layer i hidden output
            current_input = new_hx.clone();
            current_hx = new_hx;
            current_cx = new_cx;
        }

        let network_cache = LSTMNetworkCache { cell_caches };
        (current_hx, current_cx, network_cache)
    }

    /// Backward pass through all layers (reverse order)
    /// 
    /// Implements backpropagation through the multi-layer stack.
    /// Returns gradients for each layer and input gradients.
    pub fn backward(&self, dhy: &Array2<f64>, dcy: &Array2<f64>, cache: &LSTMNetworkCache) -> (Vec<LSTMCellGradients>, Array2<f64>) {
        let mut gradients = Vec::new();
        let mut current_dhy = dhy.clone();
        let mut current_dcy = dcy.clone();

        // Backward through layers in reverse order
        for (i, cell) in self.cells.iter().enumerate().rev() {
            let cell_cache = &cache.cell_caches[i];
            let (cell_gradients, dx, _dhx_prev, dcx_prev) = cell.backward(&current_dhy, &current_dcy, cell_cache);
            
            gradients.push(cell_gradients);

            // Propagate gradients to previous layer
            if i > 0 {
                current_dhy = dx;
                current_dcy = dcx_prev;
            }
        }

        gradients.reverse(); // Match forward order
        
        // Compute input gradients for the first layer
        let dx_input = if !gradients.is_empty() {
            let first_cell = &self.cells[0];
            let first_cache = &cache.cell_caches[0];
            let (_, dx_input, _, _) = first_cell.backward(dhy, dcy, first_cache);
            dx_input
        } else {
            Array2::zeros(dhy.raw_dim())
        };

        (gradients, dx_input)
    }

    /// Update parameters for all layers using computed gradients
    pub fn update_parameters<O: Optimizer>(&mut self, gradients: &[LSTMCellGradients], optimizer: &mut O) {
        for (i, (cell, cell_gradients)) in self.cells.iter_mut().zip(gradients.iter()).enumerate() {
            let prefix = format!("layer_{}", i);
            cell.update_parameters(cell_gradients, optimizer, &prefix);
        }
    }

    /// Initialize zero gradients for all layers
    pub fn zero_gradients(&self) -> Vec<LSTMCellGradients> {
        self.cells.iter().map(|cell| cell.zero_gradients()).collect()
    }

    /// Process an entire sequence with caching for training
    /// 
    /// Maintains hidden/cell state across time steps within the sequence.
    /// Returns outputs and caches for each time step.
    pub fn forward_sequence_with_cache(&self, sequence: &[Array2<f64>]) -> (Vec<(Array2<f64>, Array2<f64>)>, Vec<LSTMNetworkCache>) {
        let mut outputs = Vec::new();
        let mut caches = Vec::new();
        let mut hx = Array2::zeros((self.hidden_size, 1));
        let mut cx = Array2::zeros((self.hidden_size, 1));

        for input in sequence {
            let (new_hx, new_cx, cache) = self.forward_with_cache(input, &hx, &cx);
            outputs.push((new_hx.clone(), new_cx.clone()));
            caches.push(cache);
            hx = new_hx;
            cx = new_cx;
        }

        (outputs, caches)
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
        let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let hx = arr2(&[[0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0]]);

        let (hy, cy) = network.forward(&input, &hx, &cx);

        assert_eq!(hy.shape(), &[hidden_size, 1]);
        assert_eq!(cy.shape(), &[hidden_size, 1]);
    }
}
