use crate::layers::dropout::{zone, Dropout, Zoneout};
use crate::utils::sigmoid;
use ndarray::{s, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Holds gradients for all LSTM cell parameters during backpropagation
#[derive(Clone)]
pub struct LSTMCellGradients {
    pub w_ih: Array2<f64>,
    pub w_hh: Array2<f64>,
    pub b_ih: Array2<f64>,
    pub b_hh: Array2<f64>,
}

impl LSTMCellGradients {
    pub fn accumulate(&mut self, other: &LSTMCellGradients) {
        self.w_ih += &other.w_ih;
        self.w_hh += &other.w_hh;
        self.b_ih += &other.b_ih;
        self.b_hh += &other.b_hh;
    }

    pub fn squared_norm(&self) -> f64 {
        [&self.w_ih, &self.w_hh, &self.b_ih, &self.b_hh]
            .iter()
            .map(|m| m.iter().map(|v| v * v).sum::<f64>())
            .sum()
    }

    pub fn scale(&mut self, factor: f64) {
        for m in [
            &mut self.w_ih,
            &mut self.w_hh,
            &mut self.b_ih,
            &mut self.b_hh,
        ] {
            m.mapv_inplace(|v| v * factor);
        }
    }
}

/// Values saved by the forward pass and consumed by `backward`.
///
/// `input` and `hx` are the values after input and recurrent dropout, i.e. exactly
/// the operands of `w_ih` and `w_hh`. Matrices have one column per batch element.
#[derive(Clone)]
pub struct LSTMCellCache {
    pub input: Array2<f64>,
    pub hx: Array2<f64>,
    pub cx: Array2<f64>,
    pub input_gate: Array2<f64>,
    pub forget_gate: Array2<f64>,
    pub cell_gate: Array2<f64>,
    pub output_gate: Array2<f64>,
    pub cy: Array2<f64>,
    pub hy: Array2<f64>,
    pub input_dropout_mask: Option<Array2<f64>>,
    pub recurrent_dropout_mask: Option<Array2<f64>>,
    pub cell_zoneout_mask: Option<Array2<f64>>,
    pub hidden_zoneout_mask: Option<Array2<f64>>,
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
    /// Creates a new LSTM cell with weights drawn from U(-0.1, 0.1) and zero biases
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

    /// Clears variational dropout masks; call at the start of every sequence.
    pub fn reset_dropout_masks(&mut self) {
        for dropout in [
            &mut self.input_dropout,
            &mut self.recurrent_dropout,
            &mut self.output_dropout,
        ]
        .into_iter()
        .flatten()
        {
            dropout.reset();
        }
    }

    /// Samples the dropout mask applied to this cell's output before it is fed to
    /// the next layer. The recurrent state itself is never dropped.
    pub fn output_dropout_mask(&mut self, shape: ndarray::Dim<[usize; 2]>) -> Option<Array2<f64>> {
        self.output_dropout
            .as_mut()
            .and_then(|dropout| dropout.sample_mask(shape))
    }

    pub fn forward(
        &mut self,
        input: &Array2<f64>,
        hx: &Array2<f64>,
        cx: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let (hy, cy, _) = self.forward_with_cache(input, hx, cx);
        (hy, cy)
    }

    /// One time step for a batch of column vectors.
    ///
    /// `input` has shape (input_size, batch), `hx` and `cx` have shape
    /// (hidden_size, batch). Returns the new hidden and cell states.
    pub fn forward_with_cache(
        &mut self,
        input: &Array2<f64>,
        hx: &Array2<f64>,
        cx: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, LSTMCellCache) {
        let h = self.hidden_size;
        let input_mask = self
            .input_dropout
            .as_mut()
            .and_then(|d| d.sample_mask(input.raw_dim()));
        let recurrent_mask = self
            .recurrent_dropout
            .as_mut()
            .and_then(|d| d.sample_mask(hx.raw_dim()));
        let x = masked(input, &input_mask);
        let hx_in = masked(hx, &recurrent_mask);

        let gates = self.w_ih.dot(&x) + self.w_hh.dot(&hx_in) + &self.b_ih + &self.b_hh;
        let input_gate = gates.slice(s![0..h, ..]).mapv(sigmoid);
        let forget_gate = gates.slice(s![h..2 * h, ..]).mapv(sigmoid);
        let cell_gate = gates.slice(s![2 * h..3 * h, ..]).mapv(f64::tanh);
        let output_gate = gates.slice(s![3 * h..4 * h, ..]).mapv(sigmoid);

        let c_new = &forget_gate * cx + &input_gate * &cell_gate;
        let (cell_zoneout_mask, hidden_zoneout_mask) = match self.zoneout {
            Some(ref z) => (z.cell_mask(cx.raw_dim()), z.hidden_mask(hx.raw_dim())),
            None => (None, None),
        };
        let cy = match cell_zoneout_mask {
            Some(ref m) => zone(m, &c_new, cx),
            None => c_new,
        };
        let h_new = &output_gate * &cy.mapv(f64::tanh);
        let hy = match hidden_zoneout_mask {
            Some(ref m) => zone(m, &h_new, hx),
            None => h_new,
        };

        let cache = LSTMCellCache {
            input: x,
            hx: hx_in,
            cx: cx.clone(),
            input_gate,
            forget_gate,
            cell_gate,
            output_gate,
            cy: cy.clone(),
            hy: hy.clone(),
            input_dropout_mask: input_mask,
            recurrent_dropout_mask: recurrent_mask,
            cell_zoneout_mask,
            hidden_zoneout_mask,
        };

        (hy, cy, cache)
    }

    /// Backward pass for one time step.
    ///
    /// `dhy` and `dcy` are the gradients of the loss with respect to the hidden and
    /// cell states returned by the forward pass. Returns the parameter gradients
    /// (summed over the batch) and the gradients with respect to `input`, `hx` and `cx`.
    pub fn backward(
        &self,
        dhy: &Array2<f64>,
        dcy: &Array2<f64>,
        cache: &LSTMCellCache,
    ) -> (LSTMCellGradients, Array2<f64>, Array2<f64>, Array2<f64>) {
        let h = self.hidden_size;
        let (dh_new, dhx_zoneout) = split_zoneout(dhy, &cache.hidden_zoneout_mask);

        let tanh_cy = cache.cy.mapv(f64::tanh);
        let do_raw = &dh_new * &tanh_cy * &cache.output_gate.mapv(|o| o * (1.0 - o));
        let dc = dcy + &dh_new * &cache.output_gate * &tanh_cy.mapv(|t| 1.0 - t * t);
        let (dc_new, dcx_zoneout) = split_zoneout(&dc, &cache.cell_zoneout_mask);

        let di_raw = &dc_new * &cache.cell_gate * &cache.input_gate.mapv(|i| i * (1.0 - i));
        let df_raw = &dc_new * &cache.cx * &cache.forget_gate.mapv(|f| f * (1.0 - f));
        let dg_raw = &dc_new * &cache.input_gate * &cache.cell_gate.mapv(|g| 1.0 - g * g);

        let mut dgates = Array2::zeros((4 * h, dhy.ncols()));
        dgates.slice_mut(s![0..h, ..]).assign(&di_raw);
        dgates.slice_mut(s![h..2 * h, ..]).assign(&df_raw);
        dgates.slice_mut(s![2 * h..3 * h, ..]).assign(&dg_raw);
        dgates.slice_mut(s![3 * h..4 * h, ..]).assign(&do_raw);

        let db = dgates.sum_axis(Axis(1)).insert_axis(Axis(1));
        let gradients = LSTMCellGradients {
            w_ih: dgates.dot(&cache.input.t()),
            w_hh: dgates.dot(&cache.hx.t()),
            b_ih: db.clone(),
            b_hh: db,
        };

        let dx = masked(&self.w_ih.t().dot(&dgates), &cache.input_dropout_mask);
        let mut dhx = masked(&self.w_hh.t().dot(&dgates), &cache.recurrent_dropout_mask);
        if let Some(d) = dhx_zoneout {
            dhx += &d;
        }
        let mut dcx = &dc_new * &cache.forget_gate;
        if let Some(d) = dcx_zoneout {
            dcx += &d;
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
    pub fn update_parameters<O: crate::optimizers::Optimizer>(
        &mut self,
        gradients: &LSTMCellGradients,
        optimizer: &mut O,
        prefix: &str,
    ) {
        optimizer.update(&format!("{}_w_ih", prefix), &mut self.w_ih, &gradients.w_ih);
        optimizer.update(&format!("{}_w_hh", prefix), &mut self.w_hh, &gradients.w_hh);
        optimizer.update(&format!("{}_b_ih", prefix), &mut self.b_ih, &gradients.b_ih);
        optimizer.update(&format!("{}_b_hh", prefix), &mut self.b_hh, &gradients.b_hh);
    }
}

pub(crate) fn masked(x: &Array2<f64>, mask: &Option<Array2<f64>>) -> Array2<f64> {
    match mask {
        Some(m) => x * m,
        None => x.clone(),
    }
}

/// Splits the gradient of `m * prev + (1 - m) * new` into its `new` and `prev` parts.
pub(crate) fn split_zoneout(
    grad: &Array2<f64>,
    mask: &Option<Array2<f64>>,
) -> (Array2<f64>, Option<Array2<f64>>) {
    match mask {
        Some(m) => (grad * &m.mapv(|v| 1.0 - v), Some(grad * m)),
        None => (grad.clone(), None),
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
            .with_recurrent_dropout(0.5, false);

        let input = arr2(&[[1.0], [0.5]]);
        let hx = arr2(&[[0.1], [0.2], [0.3]]);
        let cx = arr2(&[[0.0], [0.0], [0.0]]);

        cell.train();
        let (_hy, _cy, cache) = cell.forward_with_cache(&input, &hx, &cx);

        assert!(cache.input_dropout_mask.is_some());
        assert!(cache.recurrent_dropout_mask.is_some());

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
        assert!(cache_eval.recurrent_dropout_mask.is_none());
    }
}
