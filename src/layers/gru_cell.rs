use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::utils::sigmoid;
use crate::layers::dropout::Dropout;

/// Holds gradients for all GRU cell parameters during backpropagation
#[derive(Clone)]
pub struct GRUCellGradients {
    pub w_ir: Array2<f64>,
    pub w_hr: Array2<f64>,
    pub b_ir: Array2<f64>,
    pub b_hr: Array2<f64>,
    pub w_iz: Array2<f64>,
    pub w_hz: Array2<f64>,
    pub b_iz: Array2<f64>,
    pub b_hz: Array2<f64>,
    pub w_ih: Array2<f64>,
    pub w_hh: Array2<f64>,
    pub b_ih: Array2<f64>,
    pub b_hh: Array2<f64>,
}

/// Caches intermediate values during forward pass for efficient backward computation
#[derive(Clone)]
pub struct GRUCellCache {
    pub input: Array2<f64>,
    pub hx: Array2<f64>,
    pub reset_gate: Array2<f64>,
    pub update_gate: Array2<f64>,
    pub new_gate: Array2<f64>,
    pub reset_hidden: Array2<f64>,
    pub hy: Array2<f64>,
    pub input_dropout_mask: Option<Array2<f64>>,
    pub recurrent_dropout_mask: Option<Array2<f64>>,
    pub output_dropout_mask: Option<Array2<f64>>,
}

/// GRU cell with trainable parameters and dropout support
#[derive(Clone)]
pub struct GRUCell {
    // Reset gate parameters
    pub w_ir: Array2<f64>,
    pub w_hr: Array2<f64>,
    pub b_ir: Array2<f64>,
    pub b_hr: Array2<f64>,
    
    // Update gate parameters
    pub w_iz: Array2<f64>,
    pub w_hz: Array2<f64>,
    pub b_iz: Array2<f64>,
    pub b_hz: Array2<f64>,
    
    // New gate parameters
    pub w_ih: Array2<f64>,
    pub w_hh: Array2<f64>,
    pub b_ih: Array2<f64>,
    pub b_hh: Array2<f64>,
    
    pub hidden_size: usize,
    pub input_dropout: Option<Dropout>,
    pub recurrent_dropout: Option<Dropout>,
    pub output_dropout: Option<Dropout>,
    pub is_training: bool,
}

impl GRUCell {
    /// Creates new GRU cell with Xavier-uniform weight initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let dist = Uniform::new(-0.1, 0.1);

        // Reset gate weights
        let w_ir = Array2::random((hidden_size, input_size), dist);
        let w_hr = Array2::random((hidden_size, hidden_size), dist);
        let b_ir = Array2::zeros((hidden_size, 1));
        let b_hr = Array2::zeros((hidden_size, 1));
        
        // Update gate weights
        let w_iz = Array2::random((hidden_size, input_size), dist);
        let w_hz = Array2::random((hidden_size, hidden_size), dist);
        let b_iz = Array2::zeros((hidden_size, 1));
        let b_hz = Array2::zeros((hidden_size, 1));
        
        // New gate weights
        let w_ih = Array2::random((hidden_size, input_size), dist);
        let w_hh = Array2::random((hidden_size, hidden_size), dist);
        let b_ih = Array2::zeros((hidden_size, 1));
        let b_hh = Array2::zeros((hidden_size, 1));

        GRUCell { 
            w_ir, w_hr, b_ir, b_hr,
            w_iz, w_hz, b_iz, b_hz,
            w_ih, w_hh, b_ih, b_hh,
            hidden_size,
            input_dropout: None,
            recurrent_dropout: None,
            output_dropout: None,
            is_training: true,
        }
    }

    pub fn with_input_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        if variational {
            self.input_dropout = Some(Dropout::variational(dropout_rate));
        } else {
            self.input_dropout = Some(Dropout::new(dropout_rate));
        }
        self
    }

    pub fn with_recurrent_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        if variational {
            self.recurrent_dropout = Some(Dropout::variational(dropout_rate));
        } else {
            self.recurrent_dropout = Some(Dropout::new(dropout_rate));
        }
        self
    }

    pub fn with_output_dropout(mut self, dropout_rate: f64) -> Self {
        self.output_dropout = Some(Dropout::new(dropout_rate));
        self
    }

    pub fn train(&mut self) {
        self.is_training = true;
        if let Some(ref mut dropout) = self.input_dropout {
            dropout.train();
        }
        if let Some(ref mut dropout) = self.recurrent_dropout {
            dropout.train();
        }
        if let Some(ref mut dropout) = self.output_dropout {
            dropout.train();
        }
    }

    pub fn eval(&mut self) {
        self.is_training = false;
        if let Some(ref mut dropout) = self.input_dropout {
            dropout.eval();
        }
        if let Some(ref mut dropout) = self.recurrent_dropout {
            dropout.eval();
        }
        if let Some(ref mut dropout) = self.output_dropout {
            dropout.eval();
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>, hx: &Array2<f64>) -> Array2<f64> {
        let (hy, _) = self.forward_with_cache(input, hx);
        hy
    }

    pub fn forward_with_cache(&mut self, input: &Array2<f64>, hx: &Array2<f64>) -> (Array2<f64>, GRUCellCache) {
        // Apply input dropout
        let (input_dropped, input_mask) = if let Some(ref mut dropout) = self.input_dropout {
            let dropped = dropout.forward(input);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (input.clone(), None)
        };

        // Apply recurrent dropout to hidden state
        let (hx_dropped, recurrent_mask) = if let Some(ref mut dropout) = self.recurrent_dropout {
            let dropped = dropout.forward(hx);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hx.clone(), None)
        };

        // Reset gate: r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
        let reset_gate = (&self.w_ir.dot(&input_dropped) + &self.b_ir + &self.w_hr.dot(&hx_dropped) + &self.b_hr)
            .map(|&x| sigmoid(x));

        // Update gate: z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
        let update_gate = (&self.w_iz.dot(&input_dropped) + &self.b_iz + &self.w_hz.dot(&hx_dropped) + &self.b_hz)
            .map(|&x| sigmoid(x));

        // Reset hidden state: reset_hidden = r_t ⊙ h_{t-1}
        let reset_hidden = &reset_gate * &hx_dropped;

        // New gate: h_tilde_t = tanh(W_ih * x_t + b_ih + W_hh * reset_hidden + b_hh)
        let new_gate = (&self.w_ih.dot(&input_dropped) + &self.b_ih + &self.w_hh.dot(&reset_hidden) + &self.b_hh)
            .map(|&x| x.tanh());

        // Output: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde_t
        let hy = &update_gate.map(|&x| 1.0 - x) * &hx_dropped + &update_gate * &new_gate;

        // Apply output dropout
        let (hy_final, output_mask) = if let Some(ref mut dropout) = self.output_dropout {
            let dropped = dropout.forward(&hy);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hy, None)
        };

        let cache = GRUCellCache {
            input: input.clone(),
            hx: hx.clone(),
            reset_gate: reset_gate.clone(),
            update_gate: update_gate.clone(),
            new_gate: new_gate.clone(),
            reset_hidden: reset_hidden,
            hy: hy_final.clone(),
            input_dropout_mask: input_mask,
            recurrent_dropout_mask: recurrent_mask,
            output_dropout_mask: output_mask,
        };

        (hy_final, cache)
    }

    /// Backward pass implementing GRU gradient computation with dropout
    /// 
    /// Returns (parameter_gradients, input_gradient, hidden_gradient)
    pub fn backward(&self, dhy: &Array2<f64>, cache: &GRUCellCache) -> (GRUCellGradients, Array2<f64>, Array2<f64>) {
        // Apply output dropout backward pass using saved mask
        let dhy_dropped = if let Some(ref mask) = cache.output_dropout_mask {
            let keep_prob = if let Some(ref dropout) = self.output_dropout {
                1.0 - dropout.dropout_rate
            } else {
                1.0
            };
            dhy * mask / keep_prob
        } else {
            dhy.clone()
        };

        // Gradients for output computation: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde_t
        let d_update_gate = &dhy_dropped * (&cache.new_gate - &cache.hx);
        let d_new_gate = &dhy_dropped * &cache.update_gate;
        let dhx_from_output = &dhy_dropped * cache.update_gate.map(|&x| 1.0 - x);

        // Gradients for new gate: h_tilde_t = tanh(W_ih * x_t + b_ih + W_hh * reset_hidden + b_hh)
        let d_new_gate_raw = &d_new_gate * cache.new_gate.map(|&x| 1.0 - x.powi(2));
        
        // Gradients for reset hidden: reset_hidden = r_t ⊙ h_{t-1}
        let d_reset_hidden = self.w_hh.t().dot(&d_new_gate_raw);
        let d_reset_gate = &d_reset_hidden * &cache.hx;
        let dhx_from_reset = &d_reset_hidden * &cache.reset_gate;

        // Gradients for reset gate: r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
        let d_reset_gate_raw = &d_reset_gate * &cache.reset_gate * cache.reset_gate.map(|&x| 1.0 - x);

        // Gradients for update gate: z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
        let d_update_gate_raw = &d_update_gate * &cache.update_gate * cache.update_gate.map(|&x| 1.0 - x);

        // Parameter gradients
        let dw_ir = d_reset_gate_raw.dot(&cache.input.t());
        let dw_hr = d_reset_gate_raw.dot(&cache.hx.t());
        let db_ir = d_reset_gate_raw.clone();
        let db_hr = d_reset_gate_raw.clone();

        let dw_iz = d_update_gate_raw.dot(&cache.input.t());
        let dw_hz = d_update_gate_raw.dot(&cache.hx.t());
        let db_iz = d_update_gate_raw.clone();
        let db_hz = d_update_gate_raw.clone();

        let dw_ih = d_new_gate_raw.dot(&cache.input.t());
        let dw_hh = d_new_gate_raw.dot(&cache.reset_hidden.t());
        let db_ih = d_new_gate_raw.clone();
        let db_hh = d_new_gate_raw.clone();

        let gradients = GRUCellGradients {
            w_ir: dw_ir, w_hr: dw_hr, b_ir: db_ir, b_hr: db_hr,
            w_iz: dw_iz, w_hz: dw_hz, b_iz: db_iz, b_hz: db_hz,
            w_ih: dw_ih, w_hh: dw_hh, b_ih: db_ih, b_hh: db_hh,
        };

        // Input and hidden gradients
        let mut dx = self.w_ir.t().dot(&d_reset_gate_raw) + 
                     self.w_iz.t().dot(&d_update_gate_raw) + 
                     self.w_ih.t().dot(&d_new_gate_raw);
        
        let mut dhx = dhx_from_output + dhx_from_reset + 
                      self.w_hr.t().dot(&d_reset_gate_raw) + 
                      self.w_hz.t().dot(&d_update_gate_raw);

        // Apply dropout gradients
        if let Some(ref mask) = cache.input_dropout_mask {
            let keep_prob = if let Some(ref dropout) = self.input_dropout {
                1.0 - dropout.dropout_rate
            } else {
                1.0
            };
            dx = dx * mask / keep_prob;
        }

        if let Some(ref mask) = cache.recurrent_dropout_mask {
            let keep_prob = if let Some(ref dropout) = self.recurrent_dropout {
                1.0 - dropout.dropout_rate
            } else {
                1.0
            };
            dhx = dhx * mask / keep_prob;
        }

        (gradients, dx, dhx)
    }

    /// Initialize zero gradients for accumulation
    pub fn zero_gradients(&self) -> GRUCellGradients {
        GRUCellGradients {
            w_ir: Array2::zeros(self.w_ir.raw_dim()),
            w_hr: Array2::zeros(self.w_hr.raw_dim()),
            b_ir: Array2::zeros(self.b_ir.raw_dim()),
            b_hr: Array2::zeros(self.b_hr.raw_dim()),
            w_iz: Array2::zeros(self.w_iz.raw_dim()),
            w_hz: Array2::zeros(self.w_hz.raw_dim()),
            b_iz: Array2::zeros(self.b_iz.raw_dim()),
            b_hz: Array2::zeros(self.b_hz.raw_dim()),
            w_ih: Array2::zeros(self.w_ih.raw_dim()),
            w_hh: Array2::zeros(self.w_hh.raw_dim()),
            b_ih: Array2::zeros(self.b_ih.raw_dim()),
            b_hh: Array2::zeros(self.b_hh.raw_dim()),
        }
    }

    /// Apply gradients using the provided optimizer
    pub fn update_parameters<O: crate::optimizers::Optimizer>(&mut self, gradients: &GRUCellGradients, optimizer: &mut O, prefix: &str) {
        optimizer.update(&format!("{}_w_ir", prefix), &mut self.w_ir, &gradients.w_ir);
        optimizer.update(&format!("{}_w_hr", prefix), &mut self.w_hr, &gradients.w_hr);
        optimizer.update(&format!("{}_b_ir", prefix), &mut self.b_ir, &gradients.b_ir);
        optimizer.update(&format!("{}_b_hr", prefix), &mut self.b_hr, &gradients.b_hr);
        optimizer.update(&format!("{}_w_iz", prefix), &mut self.w_iz, &gradients.w_iz);
        optimizer.update(&format!("{}_w_hz", prefix), &mut self.w_hz, &gradients.w_hz);
        optimizer.update(&format!("{}_b_iz", prefix), &mut self.b_iz, &gradients.b_iz);
        optimizer.update(&format!("{}_b_hz", prefix), &mut self.b_hz, &gradients.b_hz);
        optimizer.update(&format!("{}_w_ih", prefix), &mut self.w_ih, &gradients.w_ih);
        optimizer.update(&format!("{}_w_hh", prefix), &mut self.w_hh, &gradients.w_hh);
        optimizer.update(&format!("{}_b_ih", prefix), &mut self.b_ih, &gradients.b_ih);
        optimizer.update(&format!("{}_b_hh", prefix), &mut self.b_hh, &gradients.b_hh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_gru_cell_forward() {
        let input_size = 3;
        let hidden_size = 2;
        let mut cell = GRUCell::new(input_size, hidden_size);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let hx = arr2(&[[0.1], [0.2]]);

        let hy = cell.forward(&input, &hx);

        assert_eq!(hy.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_gru_cell_with_dropout() {
        let input_size = 3;
        let hidden_size = 2;
        let mut cell = GRUCell::new(input_size, hidden_size)
            .with_input_dropout(0.2, false)
            .with_recurrent_dropout(0.3, true)
            .with_output_dropout(0.1);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let hx = arr2(&[[0.1], [0.2]]);

        // Test training mode
        cell.train();
        let hy_train = cell.forward(&input, &hx);

        // Test evaluation mode
        cell.eval();
        let hy_eval = cell.forward(&input, &hx);

        assert_eq!(hy_train.shape(), &[hidden_size, 1]);
        assert_eq!(hy_eval.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_gru_backward_pass() {
        let input_size = 2;
        let hidden_size = 3;
        let mut cell = GRUCell::new(input_size, hidden_size);

        let input = arr2(&[[1.0], [0.5]]);
        let hx = arr2(&[[0.1], [0.2], [0.3]]);

        let (_hy, cache) = cell.forward_with_cache(&input, &hx);
        
        let dhy = arr2(&[[1.0], [1.0], [1.0]]);
        let (gradients, dx, dhx) = cell.backward(&dhy, &cache);

        assert_eq!(gradients.w_ir.shape(), &[hidden_size, input_size]);
        assert_eq!(gradients.w_hr.shape(), &[hidden_size, hidden_size]);
        assert_eq!(dx.shape(), &[input_size, 1]);
        assert_eq!(dhx.shape(), &[hidden_size, 1]);
    }
} 