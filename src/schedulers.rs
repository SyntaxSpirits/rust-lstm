use std::f64::consts::PI;

/// Learning rate scheduler trait for adaptive learning rate adjustment during training
pub trait LearningRateScheduler {
    /// Get the learning rate for the current epoch
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64;
    
    /// Reset the scheduler state (useful for multiple training runs)
    fn reset(&mut self);
    
    /// Get the name of the scheduler for logging
    fn name(&self) -> &'static str;
}

/// Constant learning rate (no scheduling)
#[derive(Clone, Debug)]
pub struct ConstantLR;

impl LearningRateScheduler for ConstantLR {
    fn get_lr(&mut self, _epoch: usize, base_lr: f64) -> f64 {
        base_lr
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "ConstantLR"
    }
}

/// Step decay scheduler: multiply LR by gamma every step_size epochs
#[derive(Clone, Debug)]
pub struct StepLR {
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f64) -> Self {
        StepLR { step_size, gamma }
    }
}

impl LearningRateScheduler for StepLR {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        let steps = epoch / self.step_size;
        base_lr * self.gamma.powi(steps as i32)
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "StepLR"
    }
}

/// Multi-step decay: multiply LR by gamma at specific milestones
#[derive(Clone, Debug)]
pub struct MultiStepLR {
    milestones: Vec<usize>,
    gamma: f64,
}

impl MultiStepLR {
    pub fn new(milestones: Vec<usize>, gamma: f64) -> Self {
        MultiStepLR { milestones, gamma }
    }
}

impl LearningRateScheduler for MultiStepLR {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        let num_reductions = self.milestones.iter()
            .filter(|&&milestone| epoch >= milestone)
            .count();
        base_lr * self.gamma.powi(num_reductions as i32)
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "MultiStepLR"
    }
}

/// Exponential decay scheduler: multiply LR by gamma every epoch
#[derive(Clone, Debug)]
pub struct ExponentialLR {
    gamma: f64,
}

impl ExponentialLR {
    pub fn new(gamma: f64) -> Self {
        ExponentialLR { gamma }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        base_lr * self.gamma.powi(epoch as i32)
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "ExponentialLR"
    }
}

/// Cosine annealing scheduler with warm restarts
#[derive(Clone, Debug)]
pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f64,
    last_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f64) -> Self {
        CosineAnnealingLR {
            t_max,
            eta_min,
            last_epoch: 0,
        }
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        self.last_epoch = epoch;
        if epoch == 0 {
            return base_lr;
        }
        
        let t = epoch % self.t_max;
        self.eta_min + (base_lr - self.eta_min) * 
            (1.0 + (PI * t as f64 / self.t_max as f64).cos()) / 2.0
    }
    
    fn reset(&mut self) {
        self.last_epoch = 0;
    }
    
    fn name(&self) -> &'static str {
        "CosineAnnealingLR"
    }
}

/// Cosine annealing with warm restarts
#[derive(Clone, Debug)]
pub struct CosineAnnealingWarmRestarts {
    t_0: usize,
    t_mult: usize,
    eta_min: f64,
    last_restart: usize,
    restart_count: usize,
}

impl CosineAnnealingWarmRestarts {
    pub fn new(t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        CosineAnnealingWarmRestarts {
            t_0,
            t_mult,
            eta_min,
            last_restart: 0,
            restart_count: 0,
        }
    }
}

impl LearningRateScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        if epoch == 0 {
            return base_lr;
        }
        
        let t_cur = epoch - self.last_restart;
        let t_i = self.t_0 * self.t_mult.pow(self.restart_count as u32);
        
        if t_cur >= t_i {
            self.last_restart = epoch;
            self.restart_count += 1;
            return base_lr;
        }
        
        self.eta_min + (base_lr - self.eta_min) * 
            (1.0 + (PI * t_cur as f64 / t_i as f64).cos()) / 2.0
    }
    
    fn reset(&mut self) {
        self.last_restart = 0;
        self.restart_count = 0;
    }
    
    fn name(&self) -> &'static str {
        "CosineAnnealingWarmRestarts"
    }
}

/// One cycle learning rate policy (popular for modern deep learning)
#[derive(Clone, Debug)]
pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    pct_start: f64,
    anneal_strategy: AnnealStrategy,
    div_factor: f64,
    final_div_factor: f64,
}

#[derive(Clone, Debug)]
pub enum AnnealStrategy {
    Cos,
    Linear,
}

impl OneCycleLR {
    pub fn new(max_lr: f64, total_steps: usize) -> Self {
        OneCycleLR {
            max_lr,
            total_steps,
            pct_start: 0.3,
            anneal_strategy: AnnealStrategy::Cos,
            div_factor: 25.0,
            final_div_factor: 10000.0,
        }
    }
    
    pub fn with_params(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        anneal_strategy: AnnealStrategy,
        div_factor: f64,
        final_div_factor: f64,
    ) -> Self {
        OneCycleLR {
            max_lr,
            total_steps,
            pct_start,
            anneal_strategy,
            div_factor,
            final_div_factor,
        }
    }
}

impl LearningRateScheduler for OneCycleLR {
    fn get_lr(&mut self, epoch: usize, _base_lr: f64) -> f64 {
        if epoch >= self.total_steps {
            return self.max_lr / self.final_div_factor;
        }
        
        let _step_ratio = epoch as f64 / self.total_steps as f64;
        let warmup_steps = (self.total_steps as f64 * self.pct_start) as usize;
        
        if epoch < warmup_steps {
            // Warmup phase
            let warmup_ratio = epoch as f64 / warmup_steps as f64;
            (self.max_lr / self.div_factor) + 
                (self.max_lr - self.max_lr / self.div_factor) * warmup_ratio
        } else {
            // Annealing phase
            let anneal_ratio = (epoch - warmup_steps) as f64 / 
                (self.total_steps - warmup_steps) as f64;
            
            match self.anneal_strategy {
                AnnealStrategy::Cos => {
                    let cos_factor = (1.0 + (PI * anneal_ratio).cos()) / 2.0;
                    (self.max_lr / self.final_div_factor) + 
                        (self.max_lr - self.max_lr / self.final_div_factor) * cos_factor
                },
                AnnealStrategy::Linear => {
                    self.max_lr - (self.max_lr - self.max_lr / self.final_div_factor) * anneal_ratio
                }
            }
        }
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "OneCycleLR"
    }
}

/// Reduce learning rate on plateau (when validation loss stops improving)
#[derive(Clone, Debug)]
pub struct ReduceLROnPlateau {
    factor: f64,
    patience: usize,
    threshold: f64,
    cooldown: usize,
    min_lr: f64,
    best_loss: f64,
    wait_count: usize,
    cooldown_counter: usize,
    current_lr: f64,
}

impl ReduceLROnPlateau {
    pub fn new(factor: f64, patience: usize) -> Self {
        ReduceLROnPlateau {
            factor,
            patience,
            threshold: 1e-4,
            cooldown: 0,
            min_lr: 0.0,
            best_loss: f64::INFINITY,
            wait_count: 0,
            cooldown_counter: 0,
            current_lr: 0.0,
        }
    }
    
    pub fn with_params(
        factor: f64,
        patience: usize,
        threshold: f64,
        cooldown: usize,
        min_lr: f64,
    ) -> Self {
        ReduceLROnPlateau {
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
            best_loss: f64::INFINITY,
            wait_count: 0,
            cooldown_counter: 0,
            current_lr: 0.0,
        }
    }
    
    /// Update the scheduler with the current validation loss
    pub fn step(&mut self, val_loss: f64, base_lr: f64) -> f64 {
        if self.current_lr == 0.0 {
            self.current_lr = base_lr;
        }
        
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return self.current_lr;
        }
        
        if val_loss < self.best_loss - self.threshold {
            self.best_loss = val_loss;
            self.wait_count = 0;
        } else {
            self.wait_count += 1;
            
            if self.wait_count >= self.patience {
                let new_lr = self.current_lr * self.factor;
                self.current_lr = new_lr.max(self.min_lr);
                self.wait_count = 0;
                self.cooldown_counter = self.cooldown;
                println!("ReduceLROnPlateau: reducing learning rate to {:.2e}", self.current_lr);
            }
        }
        
        self.current_lr
    }
}

impl LearningRateScheduler for ReduceLROnPlateau {
    fn get_lr(&mut self, _epoch: usize, base_lr: f64) -> f64 {
        if self.current_lr == 0.0 {
            self.current_lr = base_lr;
        }
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.best_loss = f64::INFINITY;
        self.wait_count = 0;
        self.cooldown_counter = 0;
        self.current_lr = 0.0;
    }
    
    fn name(&self) -> &'static str {
        "ReduceLROnPlateau"
    }
}

/// Linear learning rate schedule
#[derive(Clone, Debug)]
pub struct LinearLR {
    start_factor: f64,
    end_factor: f64,
    total_iters: usize,
}

impl LinearLR {
    pub fn new(start_factor: f64, end_factor: f64, total_iters: usize) -> Self {
        LinearLR {
            start_factor,
            end_factor,
            total_iters,
        }
    }
}

impl LearningRateScheduler for LinearLR {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        if epoch >= self.total_iters {
            return base_lr * self.end_factor;
        }
        
        let progress = epoch as f64 / self.total_iters as f64;
        let factor = self.start_factor + 
            (self.end_factor - self.start_factor) * progress;
        
        base_lr * factor
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "LinearLR"
    }
}

/// Polynomial learning rate decay
#[derive(Clone, Debug)]
pub struct PolynomialLR {
    total_iters: usize,
    power: f64,
    end_lr: f64,
}

impl PolynomialLR {
    pub fn new(total_iters: usize, power: f64, end_lr: f64) -> Self {
        PolynomialLR {
            total_iters,
            power,
            end_lr,
        }
    }
}

impl LearningRateScheduler for PolynomialLR {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        if epoch >= self.total_iters {
            return self.end_lr;
        }
        
        let factor = (1.0 - epoch as f64 / self.total_iters as f64).powf(self.power);
        self.end_lr + (base_lr - self.end_lr) * factor
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &'static str {
        "PolynomialLR"
    }
}

/// Cyclical learning rate policy with different modes
#[derive(Clone, Debug)]
pub struct CyclicalLR {
    base_lr: f64,
    max_lr: f64,
    step_size: usize,
    mode: CyclicalMode,
    gamma: f64,
    scale_mode: ScaleMode,
    last_step: usize,
}

#[derive(Clone, Debug)]
pub enum CyclicalMode {
    Triangular,
    Triangular2,
    ExpRange,
}

#[derive(Clone, Debug)]
pub enum ScaleMode {
    Cycle,
    Iterations,
}

impl CyclicalLR {
    pub fn new(base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        CyclicalLR {
            base_lr,
            max_lr,
            step_size,
            mode: CyclicalMode::Triangular,
            gamma: 1.0,
            scale_mode: ScaleMode::Cycle,
            last_step: 0,
        }
    }
    
    pub fn with_mode(mut self, mode: CyclicalMode) -> Self {
        self.mode = mode;
        self
    }
    
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }
    
    pub fn with_scale_mode(mut self, scale_mode: ScaleMode) -> Self {
        self.scale_mode = scale_mode;
        self
    }
}

impl LearningRateScheduler for CyclicalLR {
    fn get_lr(&mut self, epoch: usize, _base_lr: f64) -> f64 {
        self.last_step = epoch;
        
        let cycle = (epoch as f64 / (2.0 * self.step_size as f64)).floor() as usize;
        let x = (epoch as f64 / self.step_size as f64 - 2.0 * cycle as f64 - 1.0).abs();
        
        let scale_factor = match self.mode {
            CyclicalMode::Triangular => 1.0,
            CyclicalMode::Triangular2 => 1.0 / (2.0_f64.powi(cycle as i32 - 1)),
            CyclicalMode::ExpRange => self.gamma.powi(epoch as i32),
        };
        
        let scale_factor = match self.scale_mode {
            ScaleMode::Cycle => scale_factor,
            ScaleMode::Iterations => self.gamma.powi(epoch as i32),
        };
        
        self.base_lr + (self.max_lr - self.base_lr) * (1.0 - x).max(0.0) * scale_factor
    }
    
    fn reset(&mut self) {
        self.last_step = 0;
    }
    
    fn name(&self) -> &'static str {
        "CyclicalLR"
    }
}

/// Warmup scheduler that gradually increases learning rate
#[derive(Clone, Debug)]
pub struct WarmupScheduler<S: LearningRateScheduler> {
    warmup_epochs: usize,
    base_scheduler: S,
    warmup_start_lr: f64,
}

impl<S: LearningRateScheduler> WarmupScheduler<S> {
    pub fn new(warmup_epochs: usize, base_scheduler: S, warmup_start_lr: f64) -> Self {
        WarmupScheduler {
            warmup_epochs,
            base_scheduler,
            warmup_start_lr,
        }
    }
}

impl<S: LearningRateScheduler> LearningRateScheduler for WarmupScheduler<S> {
    fn get_lr(&mut self, epoch: usize, base_lr: f64) -> f64 {
        if epoch < self.warmup_epochs {
            // Linear warmup
            let warmup_factor = epoch as f64 / self.warmup_epochs as f64;
            self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
        } else {
            // Use base scheduler after warmup
            self.base_scheduler.get_lr(epoch - self.warmup_epochs, base_lr)
        }
    }
    
    fn reset(&mut self) {
        self.base_scheduler.reset();
    }
    
    fn name(&self) -> &'static str {
        "WarmupScheduler"
    }
}

/// Learning rate schedule visualization helper
pub struct LRScheduleVisualizer;

impl LRScheduleVisualizer {
    /// Generate learning rate values for visualization
    pub fn generate_schedule<S: LearningRateScheduler>(
        mut scheduler: S,
        base_lr: f64,
        epochs: usize,
    ) -> Vec<(usize, f64)> {
        let mut schedule = Vec::new();
        
        for epoch in 0..epochs {
            let lr = scheduler.get_lr(epoch, base_lr);
            schedule.push((epoch, lr));
        }
        
        schedule
    }
    
    /// Print ASCII visualization of learning rate schedule
    pub fn print_schedule<S: LearningRateScheduler>(
        scheduler: S,
        base_lr: f64,
        epochs: usize,
        width: usize,
        height: usize,
    ) {
        let schedule = Self::generate_schedule(scheduler, base_lr, epochs);
        
        if schedule.is_empty() {
            return;
        }
        
        let min_lr = schedule.iter().map(|(_, lr)| *lr).fold(f64::INFINITY, f64::min);
        let max_lr = schedule.iter().map(|(_, lr)| *lr).fold(0.0, f64::max);
        
        println!("Learning Rate Schedule Visualization ({}x{})", width, height);
        println!("Min LR: {:.2e}, Max LR: {:.2e}", min_lr, max_lr);
        println!("┌{}┐", "─".repeat(width));
        
        for row in 0..height {
            let y_value = max_lr - (max_lr - min_lr) * row as f64 / (height - 1) as f64;
            print!("│");
            
            for col in 0..width {
                let epoch_idx = col * epochs / width;
                let lr = if epoch_idx < schedule.len() {
                    schedule[epoch_idx].1
                } else {
                    min_lr
                };
                
                if (lr - y_value).abs() < (max_lr - min_lr) / height as f64 {
                    print!("█");
                } else {
                    print!(" ");
                }
            }
            
            println!("│ {:.2e}", y_value);
        }
        
        println!("└{}┘", "─".repeat(width));
        print!(" ");
        for i in 0..=4 {
            let epoch = i * epochs / 4;
            print!("{:>width$}", epoch, width = width / 5);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let mut scheduler = ConstantLR;
        let base_lr = 0.01;
        
        assert_eq!(scheduler.get_lr(0, base_lr), base_lr);
        assert_eq!(scheduler.get_lr(10, base_lr), base_lr);
        assert_eq!(scheduler.get_lr(100, base_lr), base_lr);
    }

    #[test]
    fn test_step_lr() {
        let mut scheduler = StepLR::new(10, 0.1);
        let base_lr = 0.01;
        
        assert_eq!(scheduler.get_lr(0, base_lr), base_lr);
        assert_eq!(scheduler.get_lr(9, base_lr), base_lr);
        assert!((scheduler.get_lr(10, base_lr) - base_lr * 0.1).abs() < 1e-15);
        assert!((scheduler.get_lr(20, base_lr) - base_lr * 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_exponential_lr() {
        let mut scheduler = ExponentialLR::new(0.9);
        let base_lr = 0.01;
        
        assert_eq!(scheduler.get_lr(0, base_lr), base_lr);
        assert!((scheduler.get_lr(1, base_lr) - base_lr * 0.9).abs() < 1e-10);
        assert!((scheduler.get_lr(2, base_lr) - base_lr * 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_multi_step_lr() {
        let mut scheduler = MultiStepLR::new(vec![10, 20], 0.1);
        let base_lr = 0.01;
        
        assert_eq!(scheduler.get_lr(5, base_lr), base_lr);
        assert!((scheduler.get_lr(10, base_lr) - base_lr * 0.1).abs() < 1e-15);
        assert!((scheduler.get_lr(15, base_lr) - base_lr * 0.1).abs() < 1e-15);
        assert!((scheduler.get_lr(20, base_lr) - base_lr * 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_one_cycle_lr() {
        let mut scheduler = OneCycleLR::new(0.1, 100);
        let base_lr = 0.01;
        
        let lr_0 = scheduler.get_lr(0, base_lr);
        let lr_30 = scheduler.get_lr(30, base_lr); // Should be close to max
        let lr_100 = scheduler.get_lr(100, base_lr); // Should be very small
        
        assert!(lr_0 < lr_30);
        assert!(lr_100 < lr_0);
        assert!(lr_30 <= 0.1);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(0.5, 2);
        let base_lr = 0.01;
        
        // Should not reduce initially
        let lr1 = scheduler.step(1.0, base_lr);
        assert_eq!(lr1, base_lr);
        
        // Should not reduce with improving loss
        let lr2 = scheduler.step(0.8, base_lr);
        assert_eq!(lr2, base_lr);
        
        // Should reduce after patience epochs without improvement
        let _lr3 = scheduler.step(0.9, base_lr);
        let _lr4 = scheduler.step(0.9, base_lr);
        let lr5 = scheduler.step(0.9, base_lr);
        
        assert!(lr5 < base_lr);
        assert!((lr5 - base_lr * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_lr() {
        let mut scheduler = LinearLR::new(1.0, 0.1, 10);
        let base_lr = 0.01;
        
        assert_eq!(scheduler.get_lr(0, base_lr), base_lr);
        assert!((scheduler.get_lr(5, base_lr) - base_lr * 0.55).abs() < 1e-10);
        assert!((scheduler.get_lr(10, base_lr) - base_lr * 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_lr() {
        let mut scheduler = PolynomialLR::new(100, 2.0, 0.01);
        let base_lr = 0.1;
        
        assert_eq!(scheduler.get_lr(0, base_lr), 0.1);
        // At epoch 50: factor = (1 - 50/100)^2 = 0.25
        // lr = 0.01 + (0.1 - 0.01) * 0.25 = 0.01 + 0.0225 = 0.0325
        assert!((scheduler.get_lr(50, base_lr) - 0.0325).abs() < 1e-10);
        assert!((scheduler.get_lr(100, base_lr) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_cyclical_lr() {
        let mut scheduler = CyclicalLR::new(0.1, 1.0, 10);
        let base_lr = 0.1;
        
        assert_eq!(scheduler.get_lr(0, base_lr), 0.1);
        // At epoch 5: cycle=0, x=0.5, lr should be at peak 
        // lr = 0.1 + (1.0 - 0.1) * (1 - 0.5) = 0.1 + 0.9 * 0.5 = 0.55
        assert!((scheduler.get_lr(5, base_lr) - 0.55).abs() < 1e-10);
        // At epoch 10: cycle=0, x=1.0, lr should be at max
        // lr = 0.1 + (1.0 - 0.1) * (1 - 1.0) = 0.1 + 0.9 * 0.0 = 0.1
        // But actually at epoch 10, we're at the peak (x=0): 0.1 + 0.9 * 1.0 = 1.0
        assert_eq!(scheduler.get_lr(10, base_lr), 1.0);
    }

    #[test]
    fn test_warmup_scheduler() {
        let base_scheduler = ConstantLR;
        let mut scheduler = WarmupScheduler::new(10, base_scheduler, 0.01);
        let base_lr = 0.1;
        
        assert_eq!(scheduler.get_lr(0, base_lr), 0.01);
        // At epoch 5: warmup_factor = 5/10 = 0.5
        // lr = 0.01 + (0.1 - 0.01) * 0.5 = 0.01 + 0.045 = 0.055
        assert!((scheduler.get_lr(5, base_lr) - 0.055).abs() < 1e-10);
        assert_eq!(scheduler.get_lr(10, base_lr), 0.1);
    }
} 