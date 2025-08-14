use ndarray::{Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::utils::sigmoid;
use crate::layers::dropout::{Dropout, Zoneout};

/// Holds gradients for all LSTM cell parameters during backpropagation
#[derive(Clone)]
pub struct LSTMCellGradients {
    pub w_ih: Array2<f64>,
    pub w_hh: Array2<f64>,
    pub b_ih: Array2<f64>,
    pub b_hh: Array2<f64>,
}

/// Caches intermediate values during forward pass for efficient backward computation
#[derive(Clone)]
pub struct LSTMCellCache {
    pub input: Array2<f64>,
    pub hx: Array2<f64>,
    pub cx: Array2<f64>,
    pub gates: Array2<f64>,
    pub input_gate: Array2<f64>,
    pub forget_gate: Array2<f64>,
    pub cell_gate: Array2<f64>,
    pub output_gate: Array2<f64>,
    pub cy: Array2<f64>,
    pub hy: Array2<f64>,
    pub input_dropout_mask: Option<Array2<f64>>,
    pub recurrent_dropout_mask: Option<Array2<f64>>,
    pub output_dropout_mask: Option<Array2<f64>>,
}

/// Batch cache for multiple sequences processed simultaneously
#[derive(Clone)]
pub struct LSTMCellBatchCache {
    pub input: Array2<f64>,
    pub hx: Array2<f64>,
    pub cx: Array2<f64>,
    pub gates: Array2<f64>,
    pub input_gate: Array2<f64>,
    pub forget_gate: Array2<f64>,
    pub cell_gate: Array2<f64>,
    pub output_gate: Array2<f64>,
    pub cy: Array2<f64>,
    pub hy: Array2<f64>,
    pub input_dropout_mask: Option<Array2<f64>>,
    pub recurrent_dropout_mask: Option<Array2<f64>>,
    pub output_dropout_mask: Option<Array2<f64>>,
    pub batch_size: usize,
}

/// LSTM cell with trainable parameters and dropout support
#[derive(Clone)]
pub struct LSTMCell {
    pub w_ih: Array2<f64>,
    pub w_hh: Array2<f64>,
    pub b_ih: Array2<f64>,
    pub b_hh: Array2<f64>,
    pub hidden_size: usize,
    pub input_dropout: Option<Dropout>,
    pub recurrent_dropout: Option<Dropout>,
    pub output_dropout: Option<Dropout>,
    pub zoneout: Option<Zoneout>,
    pub is_training: bool,
}

impl LSTMCell {
    /// Creates new LSTM cell with Xavier-uniform weight initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let dist = Uniform::new(-0.1, 0.1);

        let w_ih = Array2::random((4 * hidden_size, input_size), dist);
        let w_hh = Array2::random((4 * hidden_size, hidden_size), dist);
        let b_ih = Array2::zeros((4 * hidden_size, 1));
        let b_hh = Array2::zeros((4 * hidden_size, 1));

        LSTMCell { 
            w_ih, 
            w_hh, 
            b_ih, 
            b_hh, 
            hidden_size,
            input_dropout: None,
            recurrent_dropout: None,
            output_dropout: None,
            zoneout: None,
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

    pub fn with_zoneout(mut self, cell_zoneout_rate: f64, hidden_zoneout_rate: f64) -> Self {
        self.zoneout = Some(Zoneout::new(cell_zoneout_rate, hidden_zoneout_rate));
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
        if let Some(ref mut zoneout) = self.zoneout {
            zoneout.train();
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
        if let Some(ref mut zoneout) = self.zoneout {
            zoneout.eval();
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (hy, cy, _) = self.forward_with_cache(input, hx, cx);
        (hy, cy)
    }

    pub fn forward_with_cache(&mut self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>, LSTMCellCache) {
        let (input_dropped, input_mask) = if let Some(ref mut dropout) = self.input_dropout {
            let dropped = dropout.forward(input);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (input.clone(), None)
        };

        let (hx_dropped, recurrent_mask) = if let Some(ref mut dropout) = self.recurrent_dropout {
            let dropped = dropout.forward(hx);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hx.clone(), None)
        };

        // Compute all gates in parallel: [input_gate, forget_gate, cell_gate, output_gate]
        let gates = &self.w_ih.dot(&input_dropped) + &self.b_ih + &self.w_hh.dot(&hx_dropped) + &self.b_hh;

        let input_gate = gates.slice(s![0..self.hidden_size, ..]).map(|&x| sigmoid(x));
        let forget_gate = gates.slice(s![self.hidden_size..2*self.hidden_size, ..]).map(|&x| sigmoid(x));
        let cell_gate = gates.slice(s![2*self.hidden_size..3*self.hidden_size, ..]).map(|&x| x.tanh());
        let output_gate = gates.slice(s![3*self.hidden_size..4*self.hidden_size, ..]).map(|&x| sigmoid(x));

        let mut cy = &forget_gate * cx + &input_gate * &cell_gate;

        if let Some(ref zoneout) = self.zoneout {
            cy = zoneout.apply_cell_zoneout(&cy, cx);
        }

        let mut hy = &output_gate * cy.map(|&x| x.tanh());

        if let Some(ref zoneout) = self.zoneout {
            hy = zoneout.apply_hidden_zoneout(&hy, hx);
        }

        let (hy_final, output_mask) = if let Some(ref mut dropout) = self.output_dropout {
            let dropped = dropout.forward(&hy);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hy, None)
        };

        let cache = LSTMCellCache {
            input: input.clone(),
            hx: hx.clone(),
            cx: cx.clone(),
            gates: gates,
            input_gate: input_gate.to_owned(),
            forget_gate: forget_gate.to_owned(),
            cell_gate: cell_gate.to_owned(),
            output_gate: output_gate.to_owned(),
            cy: cy.clone(),
            hy: hy_final.clone(),
            input_dropout_mask: input_mask,
            recurrent_dropout_mask: recurrent_mask,
            output_dropout_mask: output_mask,
        };

        (hy_final, cy, cache)
    }

    /// Batch forward pass for multiple sequences simultaneously
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape (input_size, batch_size)
    /// * `hx` - Hidden state tensor of shape (hidden_size, batch_size)
    /// * `cx` - Cell state tensor of shape (hidden_size, batch_size)
    /// 
    /// # Returns
    /// * Tuple of (new_hidden_state, new_cell_state) with same batch dimensions
    pub fn forward_batch(&mut self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let batch_size = input.ncols();
        assert_eq!(hx.ncols(), batch_size, "Hidden state batch size must match input batch size");
        assert_eq!(cx.ncols(), batch_size, "Cell state batch size must match input batch size");
        assert_eq!(input.nrows(), self.w_ih.ncols(), "Input feature size must match weight matrix");
        assert_eq!(hx.nrows(), self.hidden_size, "Hidden state size must match network hidden size");
        assert_eq!(cx.nrows(), self.hidden_size, "Cell state size must match network hidden size");

        // Apply input dropout across the entire batch
        let (input_dropped, _input_mask) = if let Some(ref mut dropout) = self.input_dropout {
            let dropped = dropout.forward(input);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (input.clone(), None)
        };

        // Apply recurrent dropout across the entire batch
        let (hx_dropped, _recurrent_mask) = if let Some(ref mut dropout) = self.recurrent_dropout {
            let dropped = dropout.forward(hx);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hx.clone(), None)
        };

        // Compute all gates in parallel for the entire batch
        // gates shape: (4 * hidden_size, batch_size)
        let gates = &self.w_ih.dot(&input_dropped) + &self.b_ih.broadcast((4 * self.hidden_size, batch_size)).unwrap() 
                  + &self.w_hh.dot(&hx_dropped) + &self.b_hh.broadcast((4 * self.hidden_size, batch_size)).unwrap();

        // Extract and compute gate activations for the entire batch
        let input_gate = gates.slice(s![0..self.hidden_size, ..]).map(|&x| sigmoid(x));
        let forget_gate = gates.slice(s![self.hidden_size..2*self.hidden_size, ..]).map(|&x| sigmoid(x));
        let cell_gate = gates.slice(s![2*self.hidden_size..3*self.hidden_size, ..]).map(|&x| x.tanh());
        let output_gate = gates.slice(s![3*self.hidden_size..4*self.hidden_size, ..]).map(|&x| sigmoid(x));

        // Update cell state for entire batch
        let mut cy = &forget_gate * cx + &input_gate * &cell_gate;

        // Apply zoneout to cell state if configured
        if let Some(ref zoneout) = self.zoneout {
            for col_idx in 0..batch_size {
                let cy_col = cy.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let cx_col = cx.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let cy_zoneout = zoneout.apply_cell_zoneout(&cy_col, &cx_col);
                cy.column_mut(col_idx).assign(&cy_zoneout.column(0));
            }
        }

        // Compute hidden state for entire batch
        let mut hy = &output_gate * cy.map(|&x| x.tanh());

        // Apply zoneout to hidden state if configured
        if let Some(ref zoneout) = self.zoneout {
            for col_idx in 0..batch_size {
                let hy_col = hy.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let hx_col = hx.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let hy_zoneout = zoneout.apply_hidden_zoneout(&hy_col, &hx_col);
                hy.column_mut(col_idx).assign(&hy_zoneout.column(0));
            }
        }

        // Apply output dropout to the entire batch
        let hy_final = if let Some(ref mut dropout) = self.output_dropout {
            dropout.forward(&hy)
        } else {
            hy
        };

        (hy_final, cy)
    }

    /// Batch forward pass with caching for training
    /// 
    /// Similar to forward_batch but caches intermediate values needed for backpropagation
    pub fn forward_batch_with_cache(&mut self, input: &Array2<f64>, hx: &Array2<f64>, cx: &Array2<f64>) -> (Array2<f64>, Array2<f64>, LSTMCellBatchCache) {
        let batch_size = input.ncols();

        // Apply dropout and track masks
        let (input_dropped, input_mask) = if let Some(ref mut dropout) = self.input_dropout {
            let dropped = dropout.forward(input);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (input.clone(), None)
        };

        let (hx_dropped, recurrent_mask) = if let Some(ref mut dropout) = self.recurrent_dropout {
            let dropped = dropout.forward(hx);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hx.clone(), None)
        };

        // Compute gates for entire batch
        let gates = &self.w_ih.dot(&input_dropped) + &self.b_ih.broadcast((4 * self.hidden_size, batch_size)).unwrap()
                  + &self.w_hh.dot(&hx_dropped) + &self.b_hh.broadcast((4 * self.hidden_size, batch_size)).unwrap();

        let input_gate = gates.slice(s![0..self.hidden_size, ..]).map(|&x| sigmoid(x));
        let forget_gate = gates.slice(s![self.hidden_size..2*self.hidden_size, ..]).map(|&x| sigmoid(x));
        let cell_gate = gates.slice(s![2*self.hidden_size..3*self.hidden_size, ..]).map(|&x| x.tanh());
        let output_gate = gates.slice(s![3*self.hidden_size..4*self.hidden_size, ..]).map(|&x| sigmoid(x));

        let mut cy = &forget_gate * cx + &input_gate * &cell_gate;

        // Apply zoneout if configured
        if let Some(ref zoneout) = self.zoneout {
            for col_idx in 0..batch_size {
                let cy_col = cy.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let cx_col = cx.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let cy_zoneout = zoneout.apply_cell_zoneout(&cy_col, &cx_col);
                cy.column_mut(col_idx).assign(&cy_zoneout.column(0));
            }
        }

        let mut hy = &output_gate * cy.map(|&x| x.tanh());

        if let Some(ref zoneout) = self.zoneout {
            for col_idx in 0..batch_size {
                let hy_col = hy.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let hx_col = hx.column(col_idx).to_owned().insert_axis(ndarray::Axis(1));
                let hy_zoneout = zoneout.apply_hidden_zoneout(&hy_col, &hx_col);
                hy.column_mut(col_idx).assign(&hy_zoneout.column(0));
            }
        }

        let (hy_final, output_mask) = if let Some(ref mut dropout) = self.output_dropout {
            let dropped = dropout.forward(&hy);
            let mask = dropout.get_last_mask().map(|m| m.clone());
            (dropped, mask)
        } else {
            (hy, None)
        };

        // Create cache for backpropagation
        let cache = LSTMCellBatchCache {
            input: input.clone(),
            hx: hx.clone(),
            cx: cx.clone(),
            gates: gates.to_owned(),
            input_gate: input_gate.to_owned(),
            forget_gate: forget_gate.to_owned(),
            cell_gate: cell_gate.to_owned(),
            output_gate: output_gate.to_owned(),
            cy: cy.clone(),
            hy: hy_final.clone(),
            input_dropout_mask: input_mask,
            recurrent_dropout_mask: recurrent_mask,
            output_dropout_mask: output_mask,
            batch_size,
        };

        (hy_final, cy, cache)
    }

    /// Backward pass implementing LSTM gradient computation with dropout
    /// 
    /// Returns (parameter_gradients, input_gradient, hidden_gradient, cell_gradient)
    pub fn backward(&self, dhy: &Array2<f64>, dcy: &Array2<f64>, cache: &LSTMCellCache) -> (LSTMCellGradients, Array2<f64>, Array2<f64>, Array2<f64>) {
        let hidden_size = self.hidden_size;

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

        // Output gate gradients: ∂L/∂o_t = ∂L/∂h_t ⊙ tanh(c_t)
        let tanh_cy = cache.cy.map(|&x| x.tanh());
        let do_t = &dhy_dropped * &tanh_cy;
        let do_raw = &do_t * &cache.output_gate * (&cache.output_gate.map(|&x| 1.0 - x));

        // Cell state gradients from both tanh and direct paths
        let dcy_from_tanh = &dhy_dropped * &cache.output_gate * cache.cy.map(|&x| 1.0 - x.tanh().powi(2));
        let dcy_total = dcy + dcy_from_tanh;

        // Forget gate gradients: ∂L/∂f_t = ∂L/∂c_t ⊙ c_t-1
        let df_t = &dcy_total * &cache.cx;
        let df_raw = &df_t * &cache.forget_gate * cache.forget_gate.map(|&x| 1.0 - x);

        // Input gate gradients: ∂L/∂i_t = ∂L/∂c_t ⊙ g_t
        let di_t = &dcy_total * &cache.cell_gate;
        let di_raw = &di_t * &cache.input_gate * cache.input_gate.map(|&x| 1.0 - x);

        // Cell gate gradients: ∂L/∂g_t = ∂L/∂c_t ⊙ i_t
        let dc_t = &dcy_total * &cache.input_gate;
        let dc_raw = &dc_t * cache.cell_gate.map(|&x| 1.0 - x.powi(2));

        // Concatenate gate gradients in the same order as forward pass
        let mut dgates = Array2::zeros((4 * hidden_size, 1));
        dgates.slice_mut(s![0..hidden_size, ..]).assign(&di_raw);
        dgates.slice_mut(s![hidden_size..2*hidden_size, ..]).assign(&df_raw);
        dgates.slice_mut(s![2*hidden_size..3*hidden_size, ..]).assign(&dc_raw);
        dgates.slice_mut(s![3*hidden_size..4*hidden_size, ..]).assign(&do_raw);

        // Parameter gradients using chain rule
        let dw_ih = dgates.dot(&cache.input.t());
        let dw_hh = dgates.dot(&cache.hx.t());
        let db_ih = dgates.clone();
        let db_hh = dgates.clone();

        let gradients = LSTMCellGradients {
            w_ih: dw_ih,
            w_hh: dw_hh,
            b_ih: db_ih,
            b_hh: db_hh,
        };

        let mut dx = self.w_ih.t().dot(&dgates);
        let mut dhx = self.w_hh.t().dot(&dgates);
        let dcx = &dcy_total * &cache.forget_gate;

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

        (gradients, dx, dhx, dcx)
    }

    /// Batch backward pass for training with multiple sequences
    /// 
    /// Computes gradients for an entire batch simultaneously
    pub fn backward_batch(&self, dhy: &Array2<f64>, dcy: &Array2<f64>, cache: &LSTMCellBatchCache) -> (LSTMCellGradients, Array2<f64>, Array2<f64>, Array2<f64>) {
        let batch_size = cache.batch_size;
        let hidden_size = self.hidden_size;

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

        // Output gate gradients for entire batch
        let tanh_cy = cache.cy.map(|&x| x.tanh());
        let do_t = &dhy_dropped * &tanh_cy;
        let do_raw = &do_t * &cache.output_gate * &cache.output_gate.map(|&x| 1.0 - x);

        // Cell state gradients from both tanh and direct paths
        let dcy_from_tanh = &dhy_dropped * &cache.output_gate * cache.cy.map(|&x| 1.0 - x.tanh().powi(2));
        let dcy_total = dcy + dcy_from_tanh;

        // Gate gradients for entire batch
        let df_t = &dcy_total * &cache.cx;
        let df_raw = &df_t * &cache.forget_gate * cache.forget_gate.map(|&x| 1.0 - x);

        let di_t = &dcy_total * &cache.cell_gate;
        let di_raw = &di_t * &cache.input_gate * cache.input_gate.map(|&x| 1.0 - x);

        let dc_t = &dcy_total * &cache.input_gate;
        let dc_raw = &dc_t * cache.cell_gate.map(|&x| 1.0 - x.powi(2));

        // Concatenate gate gradients
        let mut dgates = Array2::zeros((4 * hidden_size, batch_size));
        dgates.slice_mut(s![0..hidden_size, ..]).assign(&di_raw);
        dgates.slice_mut(s![hidden_size..2*hidden_size, ..]).assign(&df_raw);
        dgates.slice_mut(s![2*hidden_size..3*hidden_size, ..]).assign(&dc_raw);
        dgates.slice_mut(s![3*hidden_size..4*hidden_size, ..]).assign(&do_raw);

        // Parameter gradients - sum across batch dimension
        let dw_ih = dgates.dot(&cache.input.t());
        let dw_hh = dgates.dot(&cache.hx.t());
        let db_ih = dgates.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        let db_hh = db_ih.clone();

        let gradients = LSTMCellGradients {
            w_ih: dw_ih,
            w_hh: dw_hh,
            b_ih: db_ih,
            b_hh: db_hh,
        };

        // Input and hidden gradients for entire batch
        let mut dx = self.w_ih.t().dot(&dgates);
        let mut dhx = self.w_hh.t().dot(&dgates);
        let dcx = &dcy_total * &cache.forget_gate;

        // Apply dropout gradients if masks exist
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

        (gradients, dx, dhx, dcx)
    }

    /// Initialize zero gradients for accumulation
    pub fn zero_gradients(&self) -> LSTMCellGradients {
        LSTMCellGradients {
            w_ih: Array2::zeros(self.w_ih.raw_dim()),
            w_hh: Array2::zeros(self.w_hh.raw_dim()),
            b_ih: Array2::zeros(self.b_ih.raw_dim()),
            b_hh: Array2::zeros(self.b_hh.raw_dim()),
        }
    }

    /// Apply gradients using the provided optimizer
    pub fn update_parameters<O: crate::optimizers::Optimizer>(&mut self, gradients: &LSTMCellGradients, optimizer: &mut O, prefix: &str) {
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
    fn test_lstm_cell_forward() {
        let input_size = 3;
        let hidden_size = 2;
        let mut cell = LSTMCell::new(input_size, hidden_size);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let hx = arr2(&[[0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0]]);

        let (hy, cy) = cell.forward(&input, &hx, &cx);

        assert_eq!(hy.shape(), &[hidden_size, 1]);
        assert_eq!(cy.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_lstm_cell_with_dropout() {
        let input_size = 3;
        let hidden_size = 2;
        let mut cell = LSTMCell::new(input_size, hidden_size)
            .with_input_dropout(0.2, false)
            .with_recurrent_dropout(0.3, true)
            .with_output_dropout(0.1)
            .with_zoneout(0.1, 0.1);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let hx = arr2(&[[0.0], [0.0]]);
        let cx = arr2(&[[0.0], [0.0]]);

        // Test training mode
        cell.train();
        let (hy_train, cy_train) = cell.forward(&input, &hx, &cx);

        // Test evaluation mode
        cell.eval();
        let (hy_eval, cy_eval) = cell.forward(&input, &hx, &cx);

        assert_eq!(hy_train.shape(), &[hidden_size, 1]);
        assert_eq!(cy_train.shape(), &[hidden_size, 1]);
        assert_eq!(hy_eval.shape(), &[hidden_size, 1]);
        assert_eq!(cy_eval.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_dropout_mask_backward_pass() {
        let input_size = 2;
        let hidden_size = 3;
        let mut cell = LSTMCell::new(input_size, hidden_size)
            .with_input_dropout(0.5, false)
            .with_output_dropout(0.5);

        let input = arr2(&[[1.0], [0.5]]);
        let hx = arr2(&[[0.1], [0.2], [0.3]]);
        let cx = arr2(&[[0.0], [0.0], [0.0]]);

        cell.train();
        let (_hy, _cy, cache) = cell.forward_with_cache(&input, &hx, &cx);

        assert!(cache.input_dropout_mask.is_some());
        assert!(cache.output_dropout_mask.is_some());

        let dhy = arr2(&[[1.0], [1.0], [1.0]]);
        let dcy = arr2(&[[0.0], [0.0], [0.0]]);
        
        let (gradients, dx, dhx, dcx) = cell.backward(&dhy, &dcy, &cache);

        assert_eq!(gradients.w_ih.shape(), &[4 * hidden_size, input_size]);
        assert_eq!(gradients.w_hh.shape(), &[4 * hidden_size, hidden_size]);
        assert_eq!(dx.shape(), &[input_size, 1]);
        assert_eq!(dhx.shape(), &[hidden_size, 1]);
        assert_eq!(dcx.shape(), &[hidden_size, 1]);

        cell.eval();
        let (_, _, cache_eval) = cell.forward_with_cache(&input, &hx, &cx);
        assert!(cache_eval.input_dropout_mask.is_none());
        assert!(cache_eval.output_dropout_mask.is_none());
    }
}
