use ndarray::Array2;
use crate::layers::lstm_cell::LSTMCell;

/// LSTM network struct containing multiple LSTM cells.
pub struct LSTMNetwork {
    cells: Vec<LSTMCell>,
}

impl LSTMNetwork {
    /// Creates a new LSTM network with the specified number of layers.
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::new();
        for _ in 0..num_layers {
            cells.push(LSTMCell::new(input_size, hidden_size));
        }
        LSTMNetwork { cells }
    }

    /// Performs a forward pass through the LSTM network.
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let hidden_size = self.cells[0].hidden_size;
        let mut hx = Array2::zeros((hidden_size, 1));
        let mut cx = Array2::zeros((hidden_size, 1));

        for cell in &self.cells {
            let (new_hx, new_cx) = cell.forward(input, &hx, &cx);
            hx = new_hx;
            cx = new_cx;
        }

        hx
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
        let output = network.forward(&input);

        assert_eq!(output.shape(), &[hidden_size, 1]);
    }
}
