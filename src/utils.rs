/// Utility functions for the LSTM library.

/// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Hyperbolic tangent activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(1000.0) > 0.99);
        assert!(sigmoid(-1000.0) < 0.01);
    }

    #[test]
    fn test_tanh() {
        assert!((tanh(0.0) - 0.0).abs() < 1e-10);
        assert!(tanh(1000.0) > 0.99);
        assert!(tanh(-1000.0) < -0.99);
    }
}
