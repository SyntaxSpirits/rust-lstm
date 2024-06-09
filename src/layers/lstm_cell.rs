use ndarray::{Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::utils::sigmoid;

/// LSTM cell struct containing the weights and biases.
pub struct LSTMCell {
    pub w_ih: Array2<f64>,
    pub w_hh: Array2<f64>,
    pub b_ih: Array2<f64>,
    pub b_hh: Array2<f64>,
    pub hidden_size: usize,
}

impl LSTMCell {
    /// Creates a new LSTM cell with random weights and biases.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let dist = Uniform::new(-0.1, 0.1);

        let w_ih = Array2::random((4 * hidden_size, input_size), dist);
        let w_hh = Array2::random((4 * hidden_size, hidden_size), dist);
        let b_ih = Array2::zeros((4 * hidden_size, 1));
        let b_hh = Array2::zeros((4 * hidden_size, 1));

        LSTMCell { w_ih, w_hh, b_ih, b_hh, hidden_size }
    }

    /// Performs a forward pass through the LSTM cell.
    pub fn forward(&self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let gates = &self.w_ih.dot(input) + &self.b_ih + &self.w_hh.dot(hx) + &self.b_hh;

        let input_gate = gates.slice(s![0..self.hidden_size, ..]).map(|&x| sigmoid(x));
        let forget_gate = gates.slice(s![self.hidden_size..2*self.hidden_size, ..]).map(|&x| sigmoid(x));
        let cell_gate = gates.slice(s![2*self.hidden_size..3*self.hidden_size, ..]).map(|&x| x.tanh());
        let output_gate = gates.slice(s![3*self.hidden_size..4*self.hidden_size, ..]).map(|&x| sigmoid(x));

        let cy = &forget_gate * cx + &input_gate * &cell_gate;
        let hy = &output_gate * cy.map(|&x| x.tanh());

        (hy, cy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_lstm_cell_forward() {
        let input_size = 3;
        let hidden_size = 2;
        let cell = LSTMCell::new(input_size, hidden_size);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let hx = arr2(&[[0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0]]);

        let (hy, cy) = cell.forward(&input, &hx, &cx);

        assert_eq!(hy.shape(), &[hidden_size, 1]);
        assert_eq!(cy.shape(), &[hidden_size, 1]);
    }
}
