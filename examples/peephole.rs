use ndarray::{arr2, Array2};
use rust_lstm::layers::peephole_lstm_cell::PeepholeLSTMCell;

fn main() {
    let input_size = 3;
    let hidden_size = 2;

    // Create Peephole cell
    let cell = PeepholeLSTMCell::new(input_size, hidden_size);

    // Single input vector: shape (3,1)
    let input: Array2<f64> = arr2(&[[0.5], [0.1], [-0.3]]);

    // Initialize hidden, cell
    let h_prev = Array2::<f64>::zeros((hidden_size, 1));
    let c_prev = Array2::<f64>::zeros((hidden_size, 1));

    // Single timestep forward pass
    let (h_t, c_t) = cell.forward(&input, &h_prev, &c_prev);

    println!("h_t = \n{:?}", h_t);
    println!("c_t = \n{:?}", c_t);
}
