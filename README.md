# Rust-LSTM

A simple LSTM (Long Short-Term Memory) neural network library implemented in Rust. This library provides basic functionalities to create and train LSTM networks.

## Features

- LSTM cell implementation
- Multi-layer LSTM network
- Random initialization of weights and biases
- Forward pass through the network

## Getting Started

### Prerequisites

Ensure you have Rust installed on your machine. If Rust is not already installed, you can install it by following the instructions on the official Rust website: https://www.rust-lang.org/tools/install.

### Installing

To use Rust-LSTM in your project, add the following to your Cargo.toml:

```toml
[dependencies]
rust-lstm = "0.1.0"
```

Then, run the following command to build your project and download the Rust-LSTM crate:

```bash
cargo build
```

## Usage

Here's a simple example demonstrating how to use the LSTM library:

```rust
use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    let input_size = 3;
    let hidden_size = 2;
    let num_layers = 2;

    // Create an LSTM network
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    // Create some example input data
    let input = Array2::from_shape_vec((input_size, 1), vec![0.5, 0.1, -0.3]).unwrap();

    // Initialize the hidden state and cell state
    let hx = Array2::zeros((hidden_size, 1));
    let cx = Array2::zeros((hidden_size, 1));

    // Perform a forward pass
    let (output, _) = network.forward(&input, &hx, &cx);

    // Print the output
    println!("Output: {:?}", output);
}

```

To run this example, save it as main.rs, and run:

```bash
cargo run
```

## Running the Tests

To run the tests included with Rust-LSTM, execute:

```bash
cargo test
```

This will run all the unit and integration tests defined in the library.

## Contributing

Contributions to Rust-LSTM are welcome! Here are a few ways you can help:

- Report bugs and issues
- Suggest new features or improvements
- Open a pull request with improvements to code or documentation
- Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
