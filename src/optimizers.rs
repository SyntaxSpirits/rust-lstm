use crate::schedulers::LearningRateScheduler;
use ndarray::Array2;
use std::collections::HashMap;

/// Optimizer trait for parameter updates during training
pub trait Optimizer {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>);
    fn reset(&mut self);

    /// Set the learning rate dynamically (for compatibility with schedulers)
    fn set_learning_rate(&mut self, lr: f64);

    /// Get the current learning rate
    fn get_learning_rate(&self) -> f64;
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

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// Adam optimizer with adaptive learning rates (Kingma & Ba, 2015).
///
/// Moments and the bias-correction step count are kept per parameter, so each
/// parameter follows the algorithm exactly however many parameters a model has.
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    state: HashMap<String, AdamState>,
}

struct AdamState {
    step: i32,
    m: Array2<f64>,
    v: Array2<f64>,
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
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        let state = self
            .state
            .entry(param_id.to_string())
            .or_insert_with(|| AdamState {
                step: 0,
                m: Array2::zeros(param.raw_dim()),
                v: Array2::zeros(param.raw_dim()),
            });
        state.step += 1;
        state.m = self.beta1 * &state.m + (1.0 - self.beta1) * gradient;
        state.v = self.beta2 * &state.v + (1.0 - self.beta2) * gradient * gradient;

        let m_hat = &state.m / (1.0 - self.beta1.powi(state.step));
        let v_hat = &state.v / (1.0 - self.beta2.powi(state.step));

        let update = self.learning_rate * m_hat / (v_hat.mapv(f64::sqrt) + self.epsilon);
        *param = &*param - update;
    }

    fn reset(&mut self) {
        self.state.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
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
            self.v
                .insert(param_id.to_string(), Array2::zeros(param.raw_dim()));
        }

        let v_t = self.v.get_mut(param_id).unwrap();

        *v_t = self.alpha * &*v_t + (1.0 - self.alpha) * gradient * gradient;

        let update = self.learning_rate * gradient / (v_t.map(|x| x.sqrt()) + self.epsilon);
        *param = &*param - update;
    }

    fn reset(&mut self) {
        self.v.clear();
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// Wrapper that combines an optimizer with a learning rate scheduler
pub struct ScheduledOptimizer<O: Optimizer, S: LearningRateScheduler> {
    optimizer: O,
    scheduler: S,
    base_lr: f64,
    current_epoch: usize,
}

impl<O: Optimizer, S: LearningRateScheduler> ScheduledOptimizer<O, S> {
    pub fn new(optimizer: O, scheduler: S, base_lr: f64) -> Self {
        ScheduledOptimizer {
            optimizer,
            scheduler,
            base_lr,
            current_epoch: 0,
        }
    }

    /// Step the scheduler (should be called at the end of each epoch)
    pub fn step(&mut self) {
        self.current_epoch += 1;
        let new_lr = self.scheduler.get_lr(self.current_epoch, self.base_lr);
        self.optimizer.set_learning_rate(new_lr);
    }

    /// Step with validation loss (for ReduceLROnPlateau)
    pub fn step_with_val_loss(&mut self, val_loss: f64) {
        self.current_epoch += 1;
        // For ReduceLROnPlateau, we need special handling
        let base_lr = self.base_lr; // Copy the value before mutable borrow
        let new_lr = if let Some(plateau_scheduler) = self.scheduler_as_plateau_mut() {
            plateau_scheduler.step(val_loss, base_lr)
        } else {
            self.scheduler.get_lr(self.current_epoch, self.base_lr)
        };
        self.optimizer.set_learning_rate(new_lr);
    }

    /// Get the current learning rate
    pub fn get_current_lr(&self) -> f64 {
        self.optimizer.get_learning_rate()
    }

    /// Get the current epoch
    pub fn get_current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Reset both optimizer and scheduler
    pub fn reset(&mut self) {
        self.optimizer.reset();
        self.scheduler.reset();
        self.current_epoch = 0;
        self.optimizer.set_learning_rate(self.base_lr);
    }

    /// Get the scheduler name for logging
    pub fn scheduler_name(&self) -> &'static str {
        self.scheduler.name()
    }

    /// Helper method to downcast scheduler to ReduceLROnPlateau if possible
    fn scheduler_as_plateau_mut(&mut self) -> Option<&mut crate::schedulers::ReduceLROnPlateau> {
        // This is a bit of a hack since we can't downcast traits easily in Rust
        // In practice, users should use step_with_val_loss only with ReduceLROnPlateau
        // For now, we'll return None and let the caller handle it properly
        None
    }
}

impl<O: Optimizer, S: LearningRateScheduler> Optimizer for ScheduledOptimizer<O, S> {
    fn update(&mut self, param_id: &str, param: &mut Array2<f64>, gradient: &Array2<f64>) {
        self.optimizer.update(param_id, param, gradient);
    }

    fn reset(&mut self) {
        self.reset(); // Call our custom reset that handles both optimizer and scheduler
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.base_lr = lr;
        self.optimizer.set_learning_rate(lr);
    }

    fn get_learning_rate(&self) -> f64 {
        self.optimizer.get_learning_rate()
    }
}

/// Helper functions to create common optimizer-scheduler combinations
impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::ConstantLR> {
    pub fn constant(optimizer: O, lr: f64) -> Self {
        Self::new(optimizer, crate::schedulers::ConstantLR, lr)
    }
}

impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::StepLR> {
    pub fn step_lr(optimizer: O, lr: f64, step_size: usize, gamma: f64) -> Self {
        Self::new(
            optimizer,
            crate::schedulers::StepLR::new(step_size, gamma),
            lr,
        )
    }
}

impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::ExponentialLR> {
    pub fn exponential(optimizer: O, lr: f64, gamma: f64) -> Self {
        Self::new(optimizer, crate::schedulers::ExponentialLR::new(gamma), lr)
    }
}

impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::CosineAnnealingLR> {
    pub fn cosine_annealing(optimizer: O, lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self::new(
            optimizer,
            crate::schedulers::CosineAnnealingLR::new(t_max, eta_min),
            lr,
        )
    }
}

impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::PolynomialLR> {
    pub fn polynomial(optimizer: O, lr: f64, total_iters: usize, power: f64, end_lr: f64) -> Self {
        Self::new(
            optimizer,
            crate::schedulers::PolynomialLR::new(total_iters, power, end_lr),
            lr,
        )
    }
}

impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::CyclicalLR> {
    pub fn cyclical(optimizer: O, base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        Self::new(
            optimizer,
            crate::schedulers::CyclicalLR::new(base_lr, max_lr, step_size),
            base_lr,
        )
    }

    pub fn cyclical_triangular2(optimizer: O, base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        let scheduler = crate::schedulers::CyclicalLR::new(base_lr, max_lr, step_size)
            .with_mode(crate::schedulers::CyclicalMode::Triangular2);
        Self::new(optimizer, scheduler, base_lr)
    }

    pub fn cyclical_exp_range(
        optimizer: O,
        base_lr: f64,
        max_lr: f64,
        step_size: usize,
        gamma: f64,
    ) -> Self {
        let scheduler = crate::schedulers::CyclicalLR::new(base_lr, max_lr, step_size)
            .with_mode(crate::schedulers::CyclicalMode::ExpRange)
            .with_gamma(gamma);
        Self::new(optimizer, scheduler, base_lr)
    }
}

impl<O: Optimizer> ScheduledOptimizer<O, crate::schedulers::OneCycleLR> {
    pub fn one_cycle(optimizer: O, max_lr: f64, total_steps: usize) -> Self {
        Self::new(
            optimizer,
            crate::schedulers::OneCycleLR::new(max_lr, total_steps),
            max_lr,
        )
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
    fn adam_steps_each_parameter_independently() {
        let lr = 0.001;
        let gradients = [
            arr2(&[[0.1, -0.2], [0.3, 0.05]]),
            arr2(&[[-0.4, 0.1], [0.2, 0.2]]),
            arr2(&[[0.3, 0.3], [-0.1, 0.6]]),
        ];

        let mut alone = Adam::new(lr);
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        for g in &gradients {
            alone.update("a", &mut a, g);
        }

        let mut shared = Adam::new(lr);
        let mut b = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut others = [Array2::zeros((3, 3)), Array2::zeros((1, 5))];
        for g in &gradients {
            for (i, other) in others.iter_mut().enumerate() {
                let og = Array2::from_elem(other.raw_dim(), 0.5);
                shared.update(&format!("other_{i}"), other, &og);
            }
            shared.update("b", &mut b, g);
        }

        assert!((&a - &b).iter().all(|d| d.abs() < 1e-15));
    }

    #[test]
    fn adam_first_step_moves_each_weight_by_the_learning_rate() {
        let mut optimizer = Adam::new(0.01);
        let mut first = arr2(&[[0.0, 0.0]]);
        let mut second = arr2(&[[0.0, 0.0]]);
        optimizer.update("first", &mut first, &arr2(&[[2.0, -0.5]]));
        optimizer.update("second", &mut second, &arr2(&[[0.3, -7.0]]));
        for value in first.iter().chain(second.iter()) {
            assert!((value.abs() - 0.01).abs() < 1e-8);
        }
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
