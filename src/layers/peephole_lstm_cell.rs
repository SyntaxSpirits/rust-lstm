use crate::optimizers::Optimizer;
use crate::random::random_array;
use crate::utils::sigmoid;
use ndarray::{Array2, Axis};
use rand_distr::Normal;

/// Gradients of every `PeepholeLSTMCell` parameter; field names match the cell.
#[derive(Clone)]
pub struct PeepholeLSTMCellGradients {
    pub w_xi: Array2<f64>,
    pub w_hi: Array2<f64>,
    pub b_i: Array2<f64>,
    pub w_ci: Array2<f64>,
    pub w_xf: Array2<f64>,
    pub w_hf: Array2<f64>,
    pub b_f: Array2<f64>,
    pub w_cf: Array2<f64>,
    pub w_xc: Array2<f64>,
    pub w_hc: Array2<f64>,
    pub b_c: Array2<f64>,
    pub w_xo: Array2<f64>,
    pub w_ho: Array2<f64>,
    pub b_o: Array2<f64>,
    pub w_co: Array2<f64>,
}

/// Values saved by the forward pass and consumed by `backward`.
#[derive(Clone)]
pub struct PeepholeLSTMCellCache {
    pub input: Array2<f64>,
    pub h_prev: Array2<f64>,
    pub c_prev: Array2<f64>,
    pub input_gate: Array2<f64>,
    pub forget_gate: Array2<f64>,
    pub cell_gate: Array2<f64>,
    pub output_gate: Array2<f64>,
    pub c: Array2<f64>,
}

/// Peephole LSTM cell with direct connections from cell state to gates
/// (Gers & Schmidhuber, 2000). The input and forget gates see the previous cell
/// state, the output gate sees the new one.
#[derive(Clone)]
pub struct PeepholeLSTMCell {
    // Input gate
    pub w_xi: Array2<f64>,
    pub w_hi: Array2<f64>,
    pub b_i: Array2<f64>,
    pub w_ci: Array2<f64>,

    // Forget gate
    pub w_xf: Array2<f64>,
    pub w_hf: Array2<f64>,
    pub b_f: Array2<f64>,
    pub w_cf: Array2<f64>,

    // Cell update
    pub w_xc: Array2<f64>,
    pub w_hc: Array2<f64>,
    pub b_c: Array2<f64>,

    // Output gate
    pub w_xo: Array2<f64>,
    pub w_ho: Array2<f64>,
    pub b_o: Array2<f64>,
    pub w_co: Array2<f64>,
}

impl PeepholeLSTMCell {
    /// Create new peephole LSTM cell with Gaussian weight initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let dist = Normal::new(0.0, 0.1).unwrap();

        let w_xi = random_array((hidden_size, input_size), dist);
        let w_hi = random_array((hidden_size, hidden_size), dist);
        let b_i = random_array((hidden_size, 1), dist);
        let w_ci = random_array((hidden_size, 1), dist);

        let w_xf = random_array((hidden_size, input_size), dist);
        let w_hf = random_array((hidden_size, hidden_size), dist);
        let b_f = random_array((hidden_size, 1), dist);
        let w_cf = random_array((hidden_size, 1), dist);

        let w_xc = random_array((hidden_size, input_size), dist);
        let w_hc = random_array((hidden_size, hidden_size), dist);
        let b_c = random_array((hidden_size, 1), dist);

        let w_xo = random_array((hidden_size, input_size), dist);
        let w_ho = random_array((hidden_size, hidden_size), dist);
        let b_o = random_array((hidden_size, 1), dist);
        let w_co = random_array((hidden_size, 1), dist);

        Self {
            w_xi,
            w_hi,
            b_i,
            w_ci,
            w_xf,
            w_hf,
            b_f,
            w_cf,
            w_xc,
            w_hc,
            b_c,
            w_xo,
            w_ho,
            b_o,
            w_co,
        }
    }

    /// Forward pass implementing peephole LSTM equations
    pub fn forward(
        &self,
        input: &Array2<f64>,
        h_prev: &Array2<f64>,
        c_prev: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let (h, c, _) = self.forward_with_cache(input, h_prev, c_prev);
        (h, c)
    }

    /// One time step for a batch of column vectors, keeping the values needed by
    /// `backward`.
    pub fn forward_with_cache(
        &self,
        input: &Array2<f64>,
        h_prev: &Array2<f64>,
        c_prev: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, PeepholeLSTMCellCache) {
        let i_t = (self.w_xi.dot(input) + self.w_hi.dot(h_prev) + &self.b_i + &self.w_ci * c_prev)
            .mapv(sigmoid);
        let f_t = (self.w_xf.dot(input) + self.w_hf.dot(h_prev) + &self.b_f + &self.w_cf * c_prev)
            .mapv(sigmoid);
        let g_t = (self.w_xc.dot(input) + self.w_hc.dot(h_prev) + &self.b_c).mapv(f64::tanh);
        let c_t = &f_t * c_prev + &i_t * &g_t;
        let o_t = (self.w_xo.dot(input) + self.w_ho.dot(h_prev) + &self.b_o + &self.w_co * &c_t)
            .mapv(sigmoid);
        let h_t = &o_t * &c_t.mapv(f64::tanh);

        let cache = PeepholeLSTMCellCache {
            input: input.clone(),
            h_prev: h_prev.clone(),
            c_prev: c_prev.clone(),
            input_gate: i_t,
            forget_gate: f_t,
            cell_gate: g_t,
            output_gate: o_t,
            c: c_t.clone(),
        };
        (h_t, c_t, cache)
    }

    /// Backward pass for one time step. Returns the parameter gradients (summed over
    /// the batch) and the gradients with respect to `input`, `h_prev` and `c_prev`.
    pub fn backward(
        &self,
        dh: &Array2<f64>,
        dc: &Array2<f64>,
        cache: &PeepholeLSTMCellCache,
    ) -> (
        PeepholeLSTMCellGradients,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
    ) {
        let (i, f, g, o) = (
            &cache.input_gate,
            &cache.forget_gate,
            &cache.cell_gate,
            &cache.output_gate,
        );
        let tanh_c = cache.c.mapv(f64::tanh);
        let do_raw = dh * &tanh_c * &o.mapv(|v| v * (1.0 - v));
        let dc_total = dc + &(dh * o * &tanh_c.mapv(|t| 1.0 - t * t)) + &do_raw * &self.w_co;
        let di_raw = &dc_total * g * &i.mapv(|v| v * (1.0 - v));
        let df_raw = &dc_total * &cache.c_prev * &f.mapv(|v| v * (1.0 - v));
        let dg_raw = &dc_total * i * &g.mapv(|v| 1.0 - v * v);

        let sum = |m: Array2<f64>| m.sum_axis(Axis(1)).insert_axis(Axis(1));
        let x_t = cache.input.t();
        let h_t = cache.h_prev.t();
        let gradients = PeepholeLSTMCellGradients {
            w_xi: di_raw.dot(&x_t),
            w_hi: di_raw.dot(&h_t),
            b_i: sum(di_raw.clone()),
            w_ci: sum(&di_raw * &cache.c_prev),
            w_xf: df_raw.dot(&x_t),
            w_hf: df_raw.dot(&h_t),
            b_f: sum(df_raw.clone()),
            w_cf: sum(&df_raw * &cache.c_prev),
            w_xc: dg_raw.dot(&x_t),
            w_hc: dg_raw.dot(&h_t),
            b_c: sum(dg_raw.clone()),
            w_xo: do_raw.dot(&x_t),
            w_ho: do_raw.dot(&h_t),
            b_o: sum(do_raw.clone()),
            w_co: sum(&do_raw * &cache.c),
        };

        let dx = self.w_xi.t().dot(&di_raw)
            + self.w_xf.t().dot(&df_raw)
            + self.w_xc.t().dot(&dg_raw)
            + self.w_xo.t().dot(&do_raw);
        let dh_prev = self.w_hi.t().dot(&di_raw)
            + self.w_hf.t().dot(&df_raw)
            + self.w_hc.t().dot(&dg_raw)
            + self.w_ho.t().dot(&do_raw);
        let dc_prev = &dc_total * f + &di_raw * &self.w_ci + &df_raw * &self.w_cf;

        (gradients, dx, dh_prev, dc_prev)
    }

    /// Apply gradients using the provided optimizer
    pub fn update_parameters<O: Optimizer>(
        &mut self,
        gradients: &PeepholeLSTMCellGradients,
        optimizer: &mut O,
        prefix: &str,
    ) {
        let params = [
            ("w_xi", &mut self.w_xi, &gradients.w_xi),
            ("w_hi", &mut self.w_hi, &gradients.w_hi),
            ("b_i", &mut self.b_i, &gradients.b_i),
            ("w_ci", &mut self.w_ci, &gradients.w_ci),
            ("w_xf", &mut self.w_xf, &gradients.w_xf),
            ("w_hf", &mut self.w_hf, &gradients.w_hf),
            ("b_f", &mut self.b_f, &gradients.b_f),
            ("w_cf", &mut self.w_cf, &gradients.w_cf),
            ("w_xc", &mut self.w_xc, &gradients.w_xc),
            ("w_hc", &mut self.w_hc, &gradients.w_hc),
            ("b_c", &mut self.b_c, &gradients.b_c),
            ("w_xo", &mut self.w_xo, &gradients.w_xo),
            ("w_ho", &mut self.w_ho, &gradients.w_ho),
            ("b_o", &mut self.b_o, &gradients.b_o),
            ("w_co", &mut self.w_co, &gradients.w_co),
        ];
        for (name, param, grad) in params {
            optimizer.update(&format!("{}_{}", prefix, name), param, grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn test_forward_shape() {
        let input_size = 3;
        let hidden_size = 2;
        let cell = PeepholeLSTMCell::new(input_size, hidden_size);

        let input = arr2(&[[0.5], [0.1], [-0.3]]);
        let h_prev = Array2::zeros((hidden_size, 1));
        let c_prev = Array2::zeros((hidden_size, 1));

        let (h_t, c_t) = cell.forward(&input, &h_prev, &c_prev);
        assert_eq!(h_t.shape(), &[hidden_size, 1]);
        assert_eq!(c_t.shape(), &[hidden_size, 1]);
    }

    #[test]
    fn test_multiple_timesteps() {
        let input_size = 3;
        let hidden_size = 2;
        let cell = PeepholeLSTMCell::new(input_size, hidden_size);

        let sequence = [
            arr2(&[[0.5], [0.1], [-0.3]]),
            arr2(&[[0.2], [0.8], [0.05]]),
            arr2(&[[0.0], [-0.1], [0.3]]),
        ];

        let mut h_prev = Array2::zeros((hidden_size, 1));
        let mut c_prev = Array2::zeros((hidden_size, 1));

        for (t, x_t) in sequence.iter().enumerate() {
            let (h_t, c_t) = cell.forward(x_t, &h_prev, &c_prev);

            assert_eq!(
                h_t.shape(),
                &[hidden_size, 1],
                "h_t shape mismatch at timestep {}",
                t
            );
            assert_eq!(
                c_t.shape(),
                &[hidden_size, 1],
                "c_t shape mismatch at timestep {}",
                t
            );

            h_prev = h_t;
            c_prev = c_t;
        }
    }
}
