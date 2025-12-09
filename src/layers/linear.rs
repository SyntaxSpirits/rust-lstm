use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::optimizers::Optimizer;

/// Holds gradients for linear layer parameters during backpropagation
#[derive(Clone, Debug)]
pub struct LinearGradients {
    pub weight: Array2<f64>,
    pub bias: Array2<f64>,
}

/// A fully connected (linear/dense) layer for neural networks
/// 
/// Performs the transformation: output = input * weight^T + bias
/// where weight has shape (output_size, input_size) and bias has shape (output_size, 1)
#[derive(Clone, Debug)]
pub struct LinearLayer {
    pub weight: Array2<f64>,     // (output_size, input_size)
    pub bias: Array2<f64>,       // (output_size, 1)
    pub input_size: usize,
    pub output_size: usize,
    input_cache: Option<Array2<f64>>, // Cache input for backward pass
}

impl LinearLayer {
    /// Create a new linear layer with random initialization
    /// 
    /// # Arguments
    /// * `input_size` - Size of input features
    /// * `output_size` - Size of output features
    /// 
    /// # Returns
    /// * New LinearLayer with Xavier/Glorot initialization
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Xavier/Glorot initialization: scale by sqrt(2 / (input_size + output_size))
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weight_range = scale;
        
        let weight = Array2::random((output_size, input_size), Uniform::new(-weight_range, weight_range));
        let bias = Array2::zeros((output_size, 1));
        
        Self {
            weight,
            bias,
            input_size,
            output_size,
            input_cache: None,
        }
    }
    
    /// Create a new linear layer with zero initialization
    pub fn new_zeros(input_size: usize, output_size: usize) -> Self {
        let weight = Array2::zeros((output_size, input_size));
        let bias = Array2::zeros((output_size, 1));
        
        Self {
            weight,
            bias,
            input_size,
            output_size,
            input_cache: None,
        }
    }
    
    /// Create a new linear layer with custom initialization
    pub fn from_weights(weight: Array2<f64>, bias: Array2<f64>) -> Self {
        let (output_size, input_size) = weight.dim();
        assert_eq!(bias.shape(), &[output_size, 1], "Bias shape must be (output_size, 1)");
        
        Self {
            weight,
            bias,
            input_size,
            output_size,
            input_cache: None,
        }
    }
    
    /// Forward pass through the linear layer
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape (input_size, batch_size)
    /// 
    /// # Returns
    /// * Output tensor of shape (output_size, batch_size)
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let (input_features, _batch_size) = input.dim();
        assert_eq!(input_features, self.input_size, 
                  "Input size {} doesn't match layer input size {}", 
                  input_features, self.input_size);
        
        // Cache input for backward pass
        self.input_cache = Some(input.clone());
        
        // output = weight @ input + bias (bias broadcasts automatically)
        &self.weight.dot(input) + &self.bias
    }
    
    /// Backward pass through the linear layer
    /// 
    /// # Arguments
    /// * `grad_output` - Gradient w.r.t. output of shape (output_size, batch_size)
    /// 
    /// # Returns
    /// * Tuple of (gradients, input_gradient)
    ///   - gradients: LinearGradients containing weight and bias gradients
    ///   - input_gradient: Gradient w.r.t. input of shape (input_size, batch_size)
    pub fn backward(&self, grad_output: &Array2<f64>) -> (LinearGradients, Array2<f64>) {
        let input = self.input_cache.as_ref().expect("Input cache not found for backward pass");
        let (output_features, batch_size) = grad_output.dim();
        let (input_features, input_batch_size) = input.dim();
        
        assert_eq!(output_features, self.output_size, "Gradient output size mismatch");
        assert_eq!(input_features, self.input_size, "Input size mismatch");
        assert_eq!(batch_size, input_batch_size, "Batch size mismatch");
        
        // Gradient w.r.t. weight: grad_output @ input^T
        let weight_grad = grad_output.dot(&input.t());
        
        // Gradient w.r.t. bias: sum over batch dimension, keep as column vector
        let bias_grad = grad_output.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        
        // Gradient w.r.t. input: weight^T @ grad_output
        let input_grad = self.weight.t().dot(grad_output);
        
        let gradients = LinearGradients {
            weight: weight_grad,
            bias: bias_grad,
        };
        
        (gradients, input_grad)
    }
    
    /// Update parameters using the provided optimizer
    pub fn update_parameters<O: Optimizer>(&mut self, gradients: &LinearGradients, optimizer: &mut O, prefix: &str) {
        optimizer.update(&format!("{}_weight", prefix), &mut self.weight, &gradients.weight);
        optimizer.update(&format!("{}_bias", prefix), &mut self.bias, &gradients.bias);
    }
    
    /// Initialize zero gradients for accumulation
    pub fn zero_gradients(&self) -> LinearGradients {
        LinearGradients {
            weight: Array2::zeros(self.weight.raw_dim()),
            bias: Array2::zeros(self.bias.raw_dim()),
        }
    }
    
    /// Get the number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
    
    /// Get layer dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.input_size, self.output_size)
    }
    
    /// Set the layer to training mode
    pub fn train(&mut self) {
        // Linear layer has no specific training mode behavior like dropout
    }
    
    /// Set the layer to evaluation mode
    pub fn eval(&mut self) {
        // Linear layer has no specific evaluation mode behavior
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::optimizers::SGD;

    #[test]
    fn test_linear_layer_creation() {
        let layer = LinearLayer::new(10, 5);
        assert_eq!(layer.input_size, 10);
        assert_eq!(layer.output_size, 5);
        assert_eq!(layer.weight.shape(), &[5, 10]);
        assert_eq!(layer.bias.shape(), &[5, 1]);
    }

    #[test]
    fn test_linear_layer_forward() {
        let mut layer = LinearLayer::new_zeros(3, 2);
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]); // (3, 2)
        
        let output = layer.forward(&input);
        assert_eq!(output.shape(), &[2, 2]); // (output_size, batch_size)
        
        // With zero weights and bias, output should be zero
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_linear_layer_backward() {
        let mut layer = LinearLayer::new(3, 2);
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]); // (3, 2)
        let grad_output = arr2(&[[1.0, 1.0], [1.0, 1.0]]); // (2, 2)
        
        // Forward pass first to cache input
        let _output = layer.forward(&input);
        
        let (gradients, input_grad) = layer.backward(&grad_output);
        
        assert_eq!(gradients.weight.shape(), &[2, 3]);
        assert_eq!(gradients.bias.shape(), &[2, 1]);
        assert_eq!(input_grad.shape(), &[3, 2]);
    }

    #[test]
    fn test_linear_layer_with_optimizer() {
        let mut layer = LinearLayer::new(2, 1);
        let mut optimizer = SGD::new(0.1);
        
        let input = arr2(&[[1.0], [2.0]]); // (2, 1)
        let target = arr2(&[[3.0]]); // (1, 1)
        
        // Forward pass
        let output = layer.forward(&input);
        
        // Simple loss gradient (output - target)
        let grad_output = &output - &target;
        
        // Backward pass
        let (gradients, _) = layer.backward(&grad_output);
        
        // Update parameters
        layer.update_parameters(&gradients, &mut optimizer, "linear");
        
        // Parameters should have changed
        assert!(layer.weight.iter().any(|&x| x != 0.0) || layer.bias.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_linear_layer_dimensions() {
        let layer = LinearLayer::new(128, 10);
        assert_eq!(layer.dimensions(), (128, 10));
        assert_eq!(layer.num_parameters(), 128 * 10 + 10); // weights + bias
    }

    #[test]
    fn test_from_weights() {
        let weight = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let bias = arr2(&[[0.5], [-0.5]]);
        
        let layer = LinearLayer::from_weights(weight.clone(), bias.clone());
        assert_eq!(layer.weight, weight);
        assert_eq!(layer.bias, bias);
        assert_eq!(layer.input_size, 2);
        assert_eq!(layer.output_size, 2);
    }
}
