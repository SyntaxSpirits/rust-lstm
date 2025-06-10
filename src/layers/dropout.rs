use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Dropout layer for regularization
/// 
/// Implements different types of dropout:
/// - Standard dropout: randomly sets elements to zero
/// - Variational dropout: uses same mask across time steps (for RNNs)
/// - Zoneout: keeps some hidden/cell state values from previous timestep
#[derive(Clone)]
pub struct Dropout {
    pub dropout_rate: f64,
    pub is_training: bool,
    pub variational: bool,
    mask: Option<Array2<f64>>,
}

impl Dropout {
    pub fn new(dropout_rate: f64) -> Self {
        assert!(dropout_rate >= 0.0 && dropout_rate <= 1.0, 
                "Dropout rate must be between 0.0 and 1.0");
        
        Dropout {
            dropout_rate,
            is_training: true,
            variational: false,
            mask: None,
        }
    }

    pub fn variational(dropout_rate: f64) -> Self {
        let mut dropout = Self::new(dropout_rate);
        dropout.variational = true;
        dropout
    }

    pub fn train(&mut self) {
        self.is_training = true;
        if self.variational {
            self.mask = None;
        }
    }

    pub fn eval(&mut self) {
        self.is_training = false;
        self.mask = None;
    }

    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        if !self.is_training || self.dropout_rate == 0.0 {
            return input.clone();
        }

        let keep_prob = 1.0 - self.dropout_rate;

        let mask = if self.variational {
            if let Some(ref mask) = self.mask {
                mask.clone()
            } else {
                let new_mask = self.generate_mask(input.raw_dim(), keep_prob);
                self.mask = Some(new_mask.clone());
                new_mask
            }
        } else {
            let new_mask = self.generate_mask(input.raw_dim(), keep_prob);
            self.mask = Some(new_mask.clone());
            new_mask
        };

        input * mask / keep_prob
    }

    pub fn get_last_mask(&self) -> Option<&Array2<f64>> {
        self.mask.as_ref()
    }

    pub fn backward(&self, grad_output: &Array2<f64>) -> Array2<f64> {
        if !self.is_training || self.dropout_rate == 0.0 {
            return grad_output.clone();
        }

        let keep_prob = 1.0 - self.dropout_rate;
        
        if let Some(ref mask) = self.mask {
            grad_output * mask / keep_prob
        } else {
            grad_output.clone()
        }
    }

    fn generate_mask(&self, shape: ndarray::Dim<[usize; 2]>, keep_prob: f64) -> Array2<f64> {
        let dist = Uniform::new(0.0, 1.0);
        Array2::random(shape, dist).mapv(|x| if x < keep_prob { 1.0 } else { 0.0 })
    }
}

/// Zoneout implementation specifically for LSTM hidden and cell states
#[derive(Clone)]
pub struct Zoneout {
    pub cell_zoneout_rate: f64,
    pub hidden_zoneout_rate: f64,
    pub is_training: bool,
}

impl Zoneout {
    pub fn new(cell_zoneout_rate: f64, hidden_zoneout_rate: f64) -> Self {
        assert!(cell_zoneout_rate >= 0.0 && cell_zoneout_rate <= 1.0);
        assert!(hidden_zoneout_rate >= 0.0 && hidden_zoneout_rate <= 1.0);
        
        Zoneout {
            cell_zoneout_rate,
            hidden_zoneout_rate,
            is_training: true,
        }
    }

    pub fn train(&mut self) {
        self.is_training = true;
    }

    pub fn eval(&mut self) {
        self.is_training = false;
    }

    pub fn apply_cell_zoneout(&self, new_cell: &Array2<f64>, prev_cell: &Array2<f64>) -> Array2<f64> {
        if !self.is_training || self.cell_zoneout_rate == 0.0 {
            return new_cell.clone();
        }

        let keep_prob = 1.0 - self.cell_zoneout_rate;
        let dist = Uniform::new(0.0, 1.0);
        let mask = Array2::random(new_cell.raw_dim(), dist);
        
        let keep_new = mask.mapv(|x| if x < keep_prob { 1.0 } else { 0.0 });
        let keep_old = mask.mapv(|x| if x >= keep_prob { 1.0 } else { 0.0 });
        
        &keep_new * new_cell + &keep_old * prev_cell
    }

    pub fn apply_hidden_zoneout(&self, new_hidden: &Array2<f64>, prev_hidden: &Array2<f64>) -> Array2<f64> {
        if !self.is_training || self.hidden_zoneout_rate == 0.0 {
            return new_hidden.clone();
        }

        let keep_prob = 1.0 - self.hidden_zoneout_rate;
        let dist = Uniform::new(0.0, 1.0);
        let mask = Array2::random(new_hidden.raw_dim(), dist);
        
        let keep_new = mask.mapv(|x| if x < keep_prob { 1.0 } else { 0.0 });
        let keep_old = mask.mapv(|x| if x >= keep_prob { 1.0 } else { 0.0 });
        
        &keep_new * new_hidden + &keep_old * prev_hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_dropout_forward() {
        let mut dropout = Dropout::new(0.5);
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        dropout.train();
        let _output_train = dropout.forward(&input);

        dropout.eval();
        let output_eval = dropout.forward(&input);
        assert_eq!(output_eval, input);
    }

    #[test]
    fn test_variational_dropout() {
        let mut dropout = Dropout::variational(0.3);
        let input1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let input2 = arr2(&[[2.0, 3.0], [4.0, 5.0]]);
        
        dropout.train();
        let _output1 = dropout.forward(&input1);
        let _output2 = dropout.forward(&input2);
    }

    #[test]
    fn test_zoneout() {
        let zoneout = Zoneout::new(0.2, 0.3);
        let new_state = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let prev_state = arr2(&[[0.5, 1.0], [1.5, 2.0]]);
        
        let result = zoneout.apply_cell_zoneout(&new_state, &prev_state);
        assert_eq!(result.shape(), new_state.shape());
    }
}