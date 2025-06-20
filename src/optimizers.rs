use ndarray::Array2;
use std::collections::HashMap;

/// Optimizer trait for parameter updates during training
pub trait Optimizer {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>);
    fn reset(&mut self);
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, _param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        *param = &*param - self.learning_rate * gradient;
    }
    
    fn reset(&mut self) {
        // SGD has no state to reset
    }
}

/// Adam optimizer with adaptive learning rates
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: i32,
    m: HashMap<String, Array2<f64>>,
    v: HashMap<String, Array2<f64>>,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Adam::with_params(learning_rate, 0.9, 0.999, 1e-8)
    }
    
    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        self.t += 1;
        
        if !self.m.contains_key(param_id) {
            self.m.insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
            self.v.insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
        }
        
        let m_t = self.m.get_mut(param_id).unwrap();
        let v_t = self.v.get_mut(param_id).unwrap();
        
        *m_t = self.beta1 * &*m_t + (1.0 - self.beta1) * gradient;
        *v_t = self.beta2 * &*v_t + (1.0 - self.beta2) * gradient * gradient;
        
        let m_hat = &*m_t / (1.0 - self.beta1.powi(self.t));
        let v_hat = &*v_t / (1.0 - self.beta2.powi(self.t));
        
        let update = self.learning_rate * m_hat / (v_hat.map(|x| x.sqrt()) + self.epsilon);
        *param = &*param - update;
    }
    
    fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

/// RMSprop optimizer
pub struct RMSprop {
    learning_rate: f64,
    alpha: f64,
    epsilon: f64,
    v: HashMap<String, Array2<f64>>,
}

impl RMSprop {
    pub fn new(learning_rate: f64) -> Self {
        RMSprop::with_params(learning_rate, 0.99, 1e-8)
    }
    
    pub fn with_params(learning_rate: f64, alpha: f64, epsilon: f64) -> Self {
        RMSprop {
            learning_rate,
            alpha,
            epsilon,
            v: HashMap::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        if !self.v.contains_key(param_id) {
            self.v.insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
        }
        
        let v_t = self.v.get_mut(param_id).unwrap();
        
        *v_t = self.alpha * &*v_t + (1.0 - self.alpha) * gradient * gradient;
        
        let update = self.learning_rate * gradient / (v_t.map(|x| x.sqrt()) + self.epsilon);
        *param = &*param - update;
    }
    
    fn reset(&mut self) {
        self.v.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGD::new(0.1);
        let mut param = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let gradient = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        
        let original_param = param.clone();
        optimizer.update("test_param", &mut param, &gradient);
        
        let expected = &original_param - 0.1 * &gradient;
        assert!((param - expected).map(|x| x.abs()).sum() < 1e-10);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::new(0.001);
        let mut param = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let gradient = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        
        let original_param = param.clone();
        optimizer.update("test_param", &mut param, &gradient);
        
        assert!((param - original_param).map(|x| x.abs()).sum() > 1e-10);
    }

    #[test]
    fn test_rmsprop_optimizer() {
        let mut optimizer = RMSprop::new(0.01);
        let mut param = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let gradient = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        
        let original_param = param.clone();
        optimizer.update("test_param", &mut param, &gradient);
        
        assert!((param - original_param).map(|x| x.abs()).sum() > 1e-10);
    }
} 