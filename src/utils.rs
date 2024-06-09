/// Utility functions for the LSTM library.

/// Sigmoid activation function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(2.0) - 0.880797).abs() < 1e-6);
        assert!((sigmoid(-2.0) - 0.1192029).abs() < 1e-6);
    }
}
