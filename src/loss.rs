use ndarray::{Array1, Array2};

/// Loss function trait for training neural networks
pub trait LossFunction {
    /// Compute the loss between predictions and targets
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;
    
    /// Compute the gradient of the loss with respect to predictions
    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64>;
}

/// Mean Squared Error loss function
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.sum() / (predictions.len() as f64)
    }
    
    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let diff = predictions - targets;
        2.0 * diff / (predictions.len() as f64)
    }
}

/// Mean Absolute Error loss function
pub struct MAELoss;

impl LossFunction for MAELoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        diff.map(|x| x.abs()).sum() / (predictions.len() as f64)
    }
    
    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let diff = predictions - targets;
        diff.map(|x| if *x > 0.0 { 1.0 } else if *x < 0.0 { -1.0 } else { 0.0 }) / (predictions.len() as f64)
    }
}

/// Cross-Entropy Loss with softmax
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let softmax_preds = softmax(predictions);
        let epsilon = 1e-15;
        let log_preds = softmax_preds.map(|x| (x + epsilon).ln());
        -(targets * log_preds).sum() / (predictions.shape()[1] as f64)
    }
    
    fn compute_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let softmax_preds = softmax(predictions);
        (softmax_preds - targets) / (predictions.shape()[1] as f64)
    }
}

/// Numerically stable softmax function
pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(x.raw_dim());
    
    for (i, col) in x.axis_iter(ndarray::Axis(1)).enumerate() {
        let max_val = col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let exp_vals: Array1<f64> = col.map(|&val| (val - max_val).exp());
        let sum_exp = exp_vals.sum();
        
        for (j, &exp_val) in exp_vals.iter().enumerate() {
            result[[j, i]] = exp_val / sum_exp;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss;
        let predictions = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let targets = arr2(&[[1.5, 2.5], [2.5, 3.5]]);
        
        let loss = loss_fn.compute_loss(&predictions, &targets);
        assert!((loss - 0.25).abs() < 1e-6);
        
        let gradient = loss_fn.compute_gradient(&predictions, &targets);
        assert_eq!(gradient.shape(), predictions.shape());
    }

    #[test]
    fn test_mae_loss() {
        let loss_fn = MAELoss;
        let predictions = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let targets = arr2(&[[1.5, 2.5], [2.5, 3.5]]);
        
        let loss = loss_fn.compute_loss(&predictions, &targets);
        assert!((loss - 0.5).abs() < 1e-6);
        
        let gradient = loss_fn.compute_gradient(&predictions, &targets);
        assert_eq!(gradient.shape(), predictions.shape());
    }

    #[test]
    fn test_softmax() {
        let input = arr2(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        let output = softmax(&input);
        
        // Each column should sum to 1
        for col in output.axis_iter(ndarray::Axis(1)) {
            let sum: f64 = col.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
} 