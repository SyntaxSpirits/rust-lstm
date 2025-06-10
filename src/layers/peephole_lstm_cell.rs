use ndarray::Array2;
use rand_distr::{Distribution, Normal};

/// Peephole LSTM cell with direct connections from cell state to gates
pub struct PeepholeLSTMCell {
    // Input gate
    pub w_xi: Array2<f64>,
    pub w_hi: Array2<f64>,
    pub b_i:  Array2<f64>,
    pub w_ci: Array2<f64>,

    // Forget gate
    pub w_xf: Array2<f64>,
    pub w_hf: Array2<f64>,
    pub b_f:  Array2<f64>,
    pub w_cf: Array2<f64>,

    // Cell update
    pub w_xc: Array2<f64>,
    pub w_hc: Array2<f64>,
    pub b_c:  Array2<f64>,

    // Output gate
    pub w_xo: Array2<f64>,
    pub w_ho: Array2<f64>,
    pub b_o:  Array2<f64>,
    pub w_co: Array2<f64>,
}

impl PeepholeLSTMCell {
    /// Create new peephole LSTM cell with Gaussian weight initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let dist = Normal::new(0.0, 0.1).unwrap();
        let mut rng = rand::thread_rng();

        let w_xi = Self::random_matrix(&dist, &mut rng, hidden_size, input_size);
        let w_hi = Self::random_matrix(&dist, &mut rng, hidden_size, hidden_size);
        let b_i = Self::random_vector_2d(&dist, &mut rng, hidden_size);
        let w_ci = Self::random_vector_2d(&dist, &mut rng, hidden_size);

        let w_xf = Self::random_matrix(&dist, &mut rng, hidden_size, input_size);
        let w_hf = Self::random_matrix(&dist, &mut rng, hidden_size, hidden_size);
        let b_f = Self::random_vector_2d(&dist, &mut rng, hidden_size);
        let w_cf = Self::random_vector_2d(&dist, &mut rng, hidden_size);

        let w_xc = Self::random_matrix(&dist, &mut rng, hidden_size, input_size);
        let w_hc = Self::random_matrix(&dist, &mut rng, hidden_size, hidden_size);
        let b_c = Self::random_vector_2d(&dist, &mut rng, hidden_size);

        let w_xo = Self::random_matrix(&dist, &mut rng, hidden_size, input_size);
        let w_ho = Self::random_matrix(&dist, &mut rng, hidden_size, hidden_size);
        let b_o = Self::random_vector_2d(&dist, &mut rng, hidden_size);
        let w_co = Self::random_vector_2d(&dist, &mut rng, hidden_size);

        Self {
            w_xi, w_hi, b_i, w_ci,
            w_xf, w_hf, b_f, w_cf,
            w_xc, w_hc, b_c,
            w_xo, w_ho, b_o, w_co,
        }
    }

    fn random_matrix(dist: &Normal<f64>, rng: &mut impl rand::Rng, rows: usize, cols: usize) -> Array2<f64> {
        let mut arr = Array2::<f64>::zeros((rows, cols));
        for val in arr.iter_mut() {
            *val = dist.sample(rng);
        }
        arr
    }

    fn random_vector_2d(dist: &Normal<f64>, rng: &mut impl rand::Rng, len: usize) -> Array2<f64> {
        let mut arr = Array2::<f64>::zeros((len, 1));
        for val in arr.iter_mut() {
            *val = dist.sample(rng);
        }
        arr
    }

    /// Forward pass implementing peephole LSTM equations
    pub fn forward(
        &self,
        input: &Array2<f64>,
        h_prev: &Array2<f64>,
        c_prev: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let i_t = &self.w_xi.dot(input)
            + &self.w_hi.dot(h_prev)
            + &self.b_i
            + &(&self.w_ci * c_prev);
        let i_t = i_t.map(|&x| sigmoid(x));

        let f_t = &self.w_xf.dot(input)
            + &self.w_hf.dot(h_prev)
            + &self.b_f
            + &(&self.w_cf * c_prev);
        let f_t = f_t.map(|&x| sigmoid(x));

        let g_t = (&self.w_xc.dot(input) + &self.w_hc.dot(h_prev) + &self.b_c)
            .map(|&x| x.tanh());

        let c_t = f_t * c_prev + i_t * g_t;

        let o_t = &self.w_xo.dot(input)
            + &self.w_ho.dot(h_prev)
            + &self.b_o
            + &(&self.w_co * &c_t);
        let o_t = o_t.map(|&x| sigmoid(x));

        let h_t = o_t * c_t.map(|&x| x.tanh());

        (h_t, c_t)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
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

        let sequence = vec![
            arr2(&[[0.5], [0.1], [-0.3]]),
            arr2(&[[0.2], [0.8], [0.05]]),
            arr2(&[[0.0], [-0.1], [0.3]]),
        ];

        let mut h_prev = Array2::zeros((hidden_size, 1));
        let mut c_prev = Array2::zeros((hidden_size, 1));

        for (t, x_t) in sequence.iter().enumerate() {
            let (h_t, c_t) = cell.forward(x_t, &h_prev, &c_prev);

            assert_eq!(h_t.shape(), &[hidden_size, 1], "h_t shape mismatch at timestep {}", t);
            assert_eq!(c_t.shape(), &[hidden_size, 1], "c_t shape mismatch at timestep {}", t);

            h_prev = h_t;
            c_prev = c_t;
        }
    }
}
