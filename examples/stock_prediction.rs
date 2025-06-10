use ndarray::{Array2, arr2};
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::MSELoss;
use rust_lstm::optimizers::Adam;

/// Stock data point with OHLCV (Open, High, Low, Close, Volume)
#[derive(Debug, Clone)]
struct StockData {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// Stock price prediction system using LSTM
struct StockPredictor {
    network: LSTMNetwork,
    trainer: Option<LSTMTrainer<MSELoss, Adam>>,
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
    sequence_length: usize,
}

impl StockPredictor {
    fn new(sequence_length: usize, hidden_size: usize) -> Self {
        // 5 features: normalized (open, high, low, close, volume)
        let network = LSTMNetwork::new(5, hidden_size, 2);
        
        Self {
            network,
            trainer: None,
            feature_means: vec![0.0; 5],
            feature_stds: vec![1.0; 5],
            sequence_length,
        }
    }

    /// Normalize features using z-score normalization
    fn fit_normalizer(&mut self, data: &[StockData]) {
        let mut sums = vec![0.0; 5];
        let mut sum_squares = vec![0.0; 5];
        let n = data.len() as f64;

        // Calculate means
        for stock in data {
            let features = [stock.open, stock.high, stock.low, stock.close, stock.volume];
            for (i, &value) in features.iter().enumerate() {
                sums[i] += value;
                sum_squares[i] += value * value;
            }
        }

        // Calculate means and standard deviations
        for i in 0..5 {
            self.feature_means[i] = sums[i] / n;
            let variance = (sum_squares[i] / n) - (self.feature_means[i] * self.feature_means[i]);
            self.feature_stds[i] = variance.sqrt().max(1e-8); // Avoid division by zero
        }
    }

    /// Normalize a single stock data point
    fn normalize_features(&self, stock: &StockData) -> Array2<f64> {
        let features = [stock.open, stock.high, stock.low, stock.close, stock.volume];
        let normalized: Vec<f64> = features.iter().enumerate()
            .map(|(i, &value)| (value - self.feature_means[i]) / self.feature_stds[i])
            .collect();
        
        Array2::from_shape_vec((5, 1), normalized).unwrap()
    }

    /// Denormalize prediction back to actual price scale
    fn denormalize_price(&self, normalized_price: f64) -> f64 {
        // Assuming we're predicting close price (index 3)
        normalized_price * self.feature_stds[3] + self.feature_means[3]
    }

    /// Create training sequences from stock data
    fn create_sequences(&self, data: &[StockData]) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
        let mut sequences = Vec::new();
        
        for i in 0..data.len().saturating_sub(self.sequence_length) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            
            // Create input sequence
            for j in i..i + self.sequence_length {
                inputs.push(self.normalize_features(&data[j]));
            }
            
            // Target sequence: predict next closing price at each timestep
            for j in i + 1..i + self.sequence_length + 1 {
                if j < data.len() {
                    let next_close = data[j].close;
                    let normalized_target = (next_close - self.feature_means[3]) / self.feature_stds[3];
                    targets.push(arr2(&[[normalized_target]]));
                }
            }
            
            // Only use sequences where inputs and targets have same length
            if inputs.len() == targets.len() && !inputs.is_empty() {
                sequences.push((inputs, targets));
            }
        }
        
        sequences
    }

    /// Train the model on historical stock data
    fn train(&mut self, data: &[StockData], validation_split: f64) {
        println!("📊 Fitting normalizer on {} data points...", data.len());
        self.fit_normalizer(data);
        
        println!("🔄 Creating training sequences...");
        let sequences = self.create_sequences(data);
        
        let split_idx = ((sequences.len() as f64) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = sequences.split_at(split_idx);
        
        println!("🎯 Training on {} sequences, validating on {} sequences", 
                 train_data.len(), val_data.len());
        
        // Create trainer with Adam optimizer
        let loss_function = MSELoss;
        let optimizer = Adam::new(0.001);
        let mut trainer = LSTMTrainer::new(self.network.clone(), loss_function, optimizer);
        
        // Train the model
        trainer.train(train_data, Some(val_data));
        
        self.trainer = Some(trainer);
        println!("✅ Training completed!");
    }

    /// Predict next day's closing price
    fn predict_next_price(&self, recent_data: &[StockData]) -> Option<f64> {
        if recent_data.len() < self.sequence_length {
            return None;
        }

        let trainer = self.trainer.as_ref()?;
        
        // Prepare input sequence
        let start_idx = recent_data.len() - self.sequence_length;
        let inputs: Vec<Array2<f64>> = recent_data[start_idx..]
            .iter()
            .map(|stock| self.normalize_features(stock))
            .collect();

        // Make prediction
        let predictions = trainer.predict(&inputs);
        
        if let Some(prediction) = predictions.last() {
            let normalized_price = prediction[[0, 0]];
            Some(self.denormalize_price(normalized_price))
        } else {
            None
        }
    }
}

/// Generate synthetic stock data for demonstration
fn generate_stock_data(days: usize) -> Vec<StockData> {
    let mut data = Vec::new();
    let mut price = 100.0;
    let volume_base = 1_000_000.0;
    
    for i in 0..days {
        // Random walk with trend and volatility
        let trend = 0.001; // Slight upward trend
        let volatility = 0.02;
        let random_change = (rand::random::<f64>() - 0.5) * volatility;
        
        price *= 1.0 + trend + random_change;
        price = price.max(1.0); // Prevent negative prices
        
        // Generate OHLC based on closing price
        let daily_volatility = 0.005;
        let high = price * (1.0 + rand::random::<f64>() * daily_volatility);
        let low = price * (1.0 - rand::random::<f64>() * daily_volatility);
        let open = low + (high - low) * rand::random::<f64>();
        
        // Volume with some correlation to price movement
        let volume_factor = 0.8 + 0.4 * rand::random::<f64>();
        let volume = volume_base * volume_factor;
        
        data.push(StockData {
            timestamp: format!("2024-01-{:02}", (i % 31) + 1),
            open,
            high,
            low,
            close: price,
            volume,
        });
    }
    
    data
}

fn main() {
    println!("🏦 Stock Price Prediction with LSTM");
    println!("=====================================\n");
    
    // Generate synthetic stock data (in practice, you'd load real data)
    let stock_data = generate_stock_data(500); // 500 days of data
    println!("📈 Generated {} days of synthetic stock data", stock_data.len());
    
    // Print sample data
    println!("\n📊 Sample data:");
    for (i, stock) in stock_data.iter().take(5).enumerate() {
        println!("Day {}: Close=${:.2}, Volume={:.0}", 
                 i + 1, stock.close, stock.volume);
    }
    
    // Create and train predictor
    let mut predictor = StockPredictor::new(20, 50); // 20-day sequences, 50 hidden units
    predictor.train(&stock_data, 0.2); // 80% train, 20% validation
    
    // Make predictions on recent data
    println!("\n🔮 Making predictions...");
    let recent_data = &stock_data[stock_data.len()-30..]; // Last 30 days
    
    for i in 20..25 { // Predict for days 21-25 of recent data
        let input_data = &recent_data[i-20..i];
        if let Some(predicted_price) = predictor.predict_next_price(input_data) {
            let actual_price = recent_data[i].close;
            let error = (predicted_price - actual_price).abs();
            let error_pct = (error / actual_price) * 100.0;
            
            println!("Day {}: Predicted=${:.2}, Actual=${:.2}, Error={:.1}%", 
                     i + 1, predicted_price, actual_price, error_pct);
        }
    }
} 