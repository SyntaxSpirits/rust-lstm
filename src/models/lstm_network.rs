use ndarray::Array2;
use crate::layers::lstm_cell::LSTMCell;

pub struct LSTMNetwork {
    cells: Vec<LSTMCell>,
}

impl LSTMNetwork {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut cells = Vec::new();
        for _ in 0..num_layers {
            cells.push(LSTMCell::new(input_size, hidden_size));
        }
        LSTMNetwork { cells }
    }

    pub fn forward(&self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let mut hx = hx.clone();
        let mut cx = cx.clone();

        for cell in &self.cells {
            let (new_hx, new_cx) = cell.forward(input, &hx, &cx);
            hx = new_hx;
            cx = new_cx;
        }

        (hx, cx)
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
