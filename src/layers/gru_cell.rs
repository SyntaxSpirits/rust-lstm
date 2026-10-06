use crate::layers::dropout::{zone, Dropout, Zoneout};
use crate::layers::lstm_cell::{masked, split_zoneout};
use crate::utils::sigmoid;
use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

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

impl GRUCellGradients {
    pub fn accumulate(&mut self, other: &GRUCellGradients) {
        for (a, b) in self.matrices_mut().into_iter().zip(other.matrices()) {
            *a += b;
        }
    }

    fn matrices(&self) -> [&Array2<f64>; 12] {
        [
            &self.w_ir, &self.w_hr, &self.b_ir, &self.b_hr, &self.w_iz, &self.w_hz, &self.b_iz,
            &self.b_hz, &self.w_ih, &self.w_hh, &self.b_ih, &self.b_hh,
        ]
    }

    fn matrices_mut(&mut self) -> [&mut Array2<f64>; 12] {
        [
            &mut self.w_ir,
            &mut self.w_hr,
            &mut self.b_ir,
            &mut self.b_hr,
            &mut self.w_iz,
            &mut self.w_hz,
            &mut self.b_iz,
            &mut self.b_hz,
            &mut self.w_ih,
            &mut self.w_hh,
            &mut self.b_ih,
            &mut self.b_hh,
        ]
    }
}

/// Values saved by the forward pass and consumed by `backward`.
///
/// `input` and `hx_dropped` are the operands of the input and recurrent weight
/// matrices after dropout; `hx` is the undropped previous state that is carried
/// through the update gate.
#[derive(Clone)]
pub struct GRUCellCache {
    pub input: Array2<f64>,
    pub hx: Array2<f64>,
    pub hx_dropped: Array2<f64>,
    pub reset_gate: Array2<f64>,
    pub update_gate: Array2<f64>,
    pub new_gate: Array2<f64>,
    pub reset_hidden: Array2<f64>,
    pub hy: Array2<f64>,
    pub input_dropout_mask: Option<Array2<f64>>,
    pub recurrent_dropout_mask: Option<Array2<f64>>,
    pub candidate_dropout_mask: Option<Array2<f64>>,
    pub zoneout_mask: Option<Array2<f64>>,
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
    pub candidate_dropout: Option<Dropout>,
    /// Zoneout of the hidden state (Krueger et al., 2017); only the hidden rate is used.
    pub zoneout: Option<Zoneout>,
    pub is_training: bool,
}

impl GRUCell {
    /// Creates a new GRU cell with weights drawn from U(-0.1, 0.1) and zero biases
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
            w_ir,
            w_hr,
            b_ir,
            b_hr,
            w_iz,
            w_hz,
            b_iz,
            b_hz,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
            input_dropout: None,
            recurrent_dropout: None,
            output_dropout: None,
            candidate_dropout: None,
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

    /// Drops units of the candidate state n_t before it is mixed into the hidden
    /// state, the GRU analogue of cell-update dropout (Semeniuta et al., 2016).
    pub fn with_candidate_dropout(mut self, dropout_rate: f64, variational: bool) -> Self {
        self.candidate_dropout = Some(if variational {
            Dropout::variational(dropout_rate)
        } else {
            Dropout::new(dropout_rate)
        });
        self
    }

    /// Each hidden unit keeps its previous value with probability `rate`.
    pub fn with_zoneout(mut self, rate: f64) -> Self {
        self.zoneout = Some(Zoneout::new(0.0, rate));
        self
    }

    fn dropouts(&mut self) -> impl Iterator<Item = &mut Dropout> {
        [
            &mut self.input_dropout,
            &mut self.recurrent_dropout,
            &mut self.output_dropout,
            &mut self.candidate_dropout,
        ]
        .into_iter()
        .flatten()
    }

    pub fn train(&mut self) {
        self.is_training = true;
        self.dropouts().for_each(Dropout::train);
        if let Some(ref mut zoneout) = self.zoneout {
            zoneout.train();
        }
    }

    pub fn eval(&mut self) {
        self.is_training = false;
        self.dropouts().for_each(Dropout::eval);
        if let Some(ref mut zoneout) = self.zoneout {
            zoneout.eval();
        }
    }

    /// Clears variational dropout masks; call at the start of every sequence.
    pub fn reset_dropout_masks(&mut self) {
        self.dropouts().for_each(Dropout::reset);
    }

    /// Samples the dropout mask applied to this cell's output before it is fed to
    /// the next layer. The recurrent state itself is never dropped.
    pub fn output_dropout_mask(&mut self, shape: ndarray::Dim<[usize; 2]>) -> Option<Array2<f64>> {
        self.output_dropout
            .as_mut()
            .and_then(|dropout| dropout.sample_mask(shape))
    }

    pub fn forward(&mut self, input: &Array2<f64>, hx: &Array2<f64>) -> Array2<f64> {
        let (hy, _) = self.forward_with_cache(input, hx);
        hy
    }

    /// One time step for a batch of column vectors (Cho et al., 2014):
    ///
    /// r = σ(W_ir x + b_ir + W_hr h + b_hr), z = σ(W_iz x + b_iz + W_hz h + b_hz),
    /// n = tanh(W_ih x + b_ih + W_hh (r ⊙ h) + b_hh), h' = (1 − z) ⊙ h + z ⊙ n.
    pub fn forward_with_cache(
        &mut self,
        input: &Array2<f64>,
        hx: &Array2<f64>,
    ) -> (Array2<f64>, GRUCellCache) {
        let input_mask = self
            .input_dropout
            .as_mut()
            .and_then(|d| d.sample_mask(input.raw_dim()));
        let recurrent_mask = self
            .recurrent_dropout
            .as_mut()
            .and_then(|d| d.sample_mask(hx.raw_dim()));
        let x = masked(input, &input_mask);
        let hd = masked(hx, &recurrent_mask);

        let reset_gate =
            (self.w_ir.dot(&x) + &self.b_ir + self.w_hr.dot(&hd) + &self.b_hr).mapv(sigmoid);
        let update_gate =
            (self.w_iz.dot(&x) + &self.b_iz + self.w_hz.dot(&hd) + &self.b_hz).mapv(sigmoid);
        let reset_hidden = &reset_gate * &hd;
        let new_gate = (self.w_ih.dot(&x) + &self.b_ih + self.w_hh.dot(&reset_hidden) + &self.b_hh)
            .mapv(f64::tanh);
        let candidate_mask = self
            .candidate_dropout
            .as_mut()
            .and_then(|d| d.sample_mask(hx.raw_dim()));
        let h_new = &update_gate.mapv(|z| 1.0 - z) * hx
            + &update_gate * &masked(&new_gate, &candidate_mask);
        let zoneout_mask = self
            .zoneout
            .as_ref()
            .and_then(|z| z.hidden_mask(hx.raw_dim()));
        let hy = match zoneout_mask {
            Some(ref m) => zone(m, &h_new, hx),
            None => h_new,
        };

        let cache = GRUCellCache {
            input: x,
            hx: hx.clone(),
            hx_dropped: hd,
            reset_gate,
            update_gate,
            new_gate,
            reset_hidden,
            hy: hy.clone(),
            input_dropout_mask: input_mask,
            recurrent_dropout_mask: recurrent_mask,
            candidate_dropout_mask: candidate_mask,
            zoneout_mask,
        };
        (hy, cache)
    }

    /// Backward pass for one time step.
    ///
    /// Returns the parameter gradients (summed over the batch) and the gradients with
    /// respect to `input` and `hx`.
    pub fn backward(
        &self,
        dhy: &Array2<f64>,
        cache: &GRUCellCache,
    ) -> (GRUCellGradients, Array2<f64>, Array2<f64>) {
        let z = &cache.update_gate;
        let r = &cache.reset_gate;
        let n = &cache.new_gate;

        let (dhy, dhx_zoneout) = split_zoneout(dhy, &cache.zoneout_mask);
        let dhy = &dhy;
        let mask = &cache.candidate_dropout_mask;
        let dz_raw = dhy * &(masked(n, mask) - &cache.hx) * &z.mapv(|v| v * (1.0 - v));
        let dn_raw = masked(&(dhy * z), mask) * &n.mapv(|v| 1.0 - v * v);
        let d_reset_hidden = self.w_hh.t().dot(&dn_raw);
        let dr_raw = &d_reset_hidden * &cache.hx_dropped * &r.mapv(|v| v * (1.0 - v));

        let sum = |m: &Array2<f64>| m.sum_axis(Axis(1)).insert_axis(Axis(1));
        let gradients = GRUCellGradients {
            w_ir: dr_raw.dot(&cache.input.t()),
            w_hr: dr_raw.dot(&cache.hx_dropped.t()),
            b_ir: sum(&dr_raw),
            b_hr: sum(&dr_raw),
            w_iz: dz_raw.dot(&cache.input.t()),
            w_hz: dz_raw.dot(&cache.hx_dropped.t()),
            b_iz: sum(&dz_raw),
            b_hz: sum(&dz_raw),
            w_ih: dn_raw.dot(&cache.input.t()),
            w_hh: dn_raw.dot(&cache.reset_hidden.t()),
            b_ih: sum(&dn_raw),
            b_hh: sum(&dn_raw),
        };

        let dx =
            self.w_ir.t().dot(&dr_raw) + self.w_iz.t().dot(&dz_raw) + self.w_ih.t().dot(&dn_raw);
        let dx = masked(&dx, &cache.input_dropout_mask);
        let dhd = &d_reset_hidden * r + self.w_hr.t().dot(&dr_raw) + self.w_hz.t().dot(&dz_raw);
        let mut dhx = dhy * &z.mapv(|v| 1.0 - v) + masked(&dhd, &cache.recurrent_dropout_mask);
        if let Some(d) = dhx_zoneout {
            dhx += &d;
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
    pub fn update_parameters<O: crate::optimizers::Optimizer>(
        &mut self,
        gradients: &GRUCellGradients,
        optimizer: &mut O,
        prefix: &str,
    ) {
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
