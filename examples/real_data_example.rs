use ndarray::{Array2, arr2};
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::MSELoss;
use rust_lstm::optimizers::Adam;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Generic data point for time series
#[derive(Debug, Clone)]
struct DataPoint {
    timestamp: String,
    values: Vec<f64>,
}

/// Data loader for CSV files
struct CSVDataLoader {
    data: Vec<DataPoint>,
    feature_names: Vec<String>,
    normalizers: Vec<(f64, f64)>, // (mean, std) for each feature
}

impl CSVDataLoader {
    /// Load data from CSV file with headers
    #[allow(dead_code)]
    fn from_csv(file_path: &str, target_column: &str) -> std::io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Read header
        let header_line = lines.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty file")
        })??;
        
        let headers: Vec<String> = header_line.split(',')
            .map(|s| s.trim().to_string())
            .collect();
        
        // Find target column index
        let _target_idx = headers.iter().position(|h| h == target_column)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, 
                                   format!("Target column '{}' not found", target_column))
            })?;
        
        let mut data = Vec::new();
        
        // Read data rows
        for line in lines {
            let line = line?;
            let values: Result<Vec<f64>, _> = line.split(',')
                .enumerate()
                .filter_map(|(i, s)| {
                    if i == 0 { None } // Skip timestamp column
                    else { Some(s.trim().parse::<f64>()) }
                })
                .collect();
            
            match values {
                Ok(vals) if !vals.is_empty() => {
                    let timestamp = line.split(',').next().unwrap_or("").to_string();
                    data.push(DataPoint { timestamp, values: vals });
                },
                _ => continue, // Skip invalid rows
            }
        }
        
        let feature_names = headers[1..].to_vec(); // Skip timestamp
        
        Ok(Self {
            data,
            feature_names,
            normalizers: Vec::new(),
        })
    }
    
    /// Generate synthetic CSV-like data for demonstration
    fn generate_synthetic_sensor_data(days: usize) -> Self {
        let mut data = Vec::new();
        
        // Simulate IoT sensor data: temperature, humidity, pressure, light
        for i in 0..days * 24 { // Hourly data
            let hour_of_day = (i % 24) as f64;
            let day_of_year = (i / 24 % 365) as f64;
            
            // Temperature with daily and seasonal cycles
            let daily_temp_cycle = 5.0 * (2.0 * std::f64::consts::PI * hour_of_day / 24.0).cos();
            let seasonal_temp_cycle = 15.0 * (2.0 * std::f64::consts::PI * day_of_year / 365.0).sin();
            let temperature = 20.0 + daily_temp_cycle + seasonal_temp_cycle + 
                            (rand::random::<f64>() - 0.5) * 3.0;
            
            // Humidity inversely related to temperature
            let humidity = 70.0 - (temperature - 20.0) * 1.5 + 
                          (rand::random::<f64>() - 0.5) * 15.0;
            let humidity = humidity.clamp(20.0, 95.0);
            
            // Pressure with weather patterns
            let pressure = 1013.25 + 10.0 * (day_of_year / 30.0).sin() + 
                          (rand::random::<f64>() - 0.5) * 20.0;
            
            // Light with daily cycle
            let light = if hour_of_day >= 6.0 && hour_of_day <= 18.0 {
                1000.0 * (std::f64::consts::PI * (hour_of_day - 6.0) / 12.0).sin() + 
                (rand::random::<f64>() - 0.5) * 200.0
            } else {
                (rand::random::<f64>() * 50.0).max(0.0)
            };
            
            let timestamp = format!("2024-{:03}-{:02}", day_of_year as u32 + 1, hour_of_day as u32);
            data.push(DataPoint {
                timestamp,
                values: vec![temperature, humidity, pressure, light],
            });
        }
        
        Self {
            data,
            feature_names: vec![
                "temperature".to_string(),
                "humidity".to_string(), 
                "pressure".to_string(),
                "light".to_string()
            ],
            normalizers: Vec::new(),
        }
    }
    
    /// Fit normalizers for all features
    fn fit_normalizers(&mut self) {
        let num_features = self.feature_names.len();
        let mut sums = vec![0.0; num_features];
        let mut sum_squares = vec![0.0; num_features];
        let n = self.data.len() as f64;
        
        // Calculate means and variances
        for point in &self.data {
            for (i, &value) in point.values.iter().enumerate() {
                sums[i] += value;
                sum_squares[i] += value * value;
            }
        }
        
        self.normalizers = sums.iter().enumerate()
            .map(|(i, &sum)| {
                let mean = sum / n;
                let variance = (sum_squares[i] / n) - (mean * mean);
                let std = variance.sqrt().max(1e-8);
                (mean, std)
            })
            .collect();
    }
    
    /// Normalize a data point
    fn normalize(&self, point: &DataPoint) -> Array2<f64> {
        let normalized: Vec<f64> = point.values.iter().enumerate()
            .map(|(i, &value)| {
                let (mean, std) = self.normalizers[i];
                (value - mean) / std
            })
            .collect();
        
        Array2::from_shape_vec((normalized.len(), 1), normalized).unwrap()
    }
    
    /// Denormalize a prediction (for first feature)
    fn denormalize(&self, normalized_value: f64, feature_idx: usize) -> f64 {
        let (mean, std) = self.normalizers[feature_idx];
        normalized_value * std + mean
    }
}

/// Time series prediction system
struct TimeSeriesPredictor {
    network: LSTMNetwork,
    trainer: Option<LSTMTrainer<MSELoss, Adam>>,
    sequence_length: usize,
    target_feature: usize,
}

impl TimeSeriesPredictor {
    fn new(input_features: usize, sequence_length: usize, hidden_size: usize, target_feature: usize) -> Self {
        // Create network: input_features -> hidden_size -> 1 output (single layer)
        let network = LSTMNetwork::new(input_features, hidden_size, 1);
        
        Self {
            network,
            trainer: None,
            sequence_length,
            target_feature,
        }
    }
    
    /// Create training sequences from data
    fn create_sequences(&self, data_loader: &CSVDataLoader) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
        let mut sequences = Vec::new();
        
        for i in 0..data_loader.data.len().saturating_sub(self.sequence_length) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            
            // Input sequence and corresponding target sequence
            for j in i..i + self.sequence_length {
                inputs.push(data_loader.normalize(&data_loader.data[j]));
                
                // Target: next value of target feature for each time step
                if j + 1 < data_loader.data.len() {
                    let next_point = &data_loader.data[j + 1];
                    let target_value = next_point.values[self.target_feature];
                    let (mean, std) = data_loader.normalizers[self.target_feature];
                    let normalized_target = (target_value - mean) / std;
                    targets.push(arr2(&[[normalized_target]])); // Match network output size (hidden_size, 1)
                }
            }
            
            if inputs.len() == targets.len() && !inputs.is_empty() {
                sequences.push((inputs, targets));
            }
        }
        
        sequences
    }
    
    /// Train the prediction model
    fn train(&mut self, data_loader: &CSVDataLoader, validation_split: f64) {
        println!("📊 Creating training sequences...");
        let sequences = self.create_sequences(data_loader);
        
        let split_idx = ((sequences.len() as f64) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = sequences.split_at(split_idx);
        
        println!("🎯 Training on {} sequences, validating on {} sequences",
                train_data.len(), val_data.len());
        
        let loss_function = MSELoss;
        let optimizer = Adam::new(0.001);
        let mut trainer = LSTMTrainer::new(self.network.clone(), loss_function, optimizer);
        
        // Configure for quick demo
        let mut config = rust_lstm::training::TrainingConfig::default();
        config.epochs = 5; // Very reduced for quick demo
        config.print_every = 2; // Print every 2 epochs
        
        trainer = trainer.with_config(config);
        
        trainer.train(train_data, Some(val_data));
        
        self.trainer = Some(trainer);
        println!("✅ Time series model training completed!");
    }
    
    /// Make prediction for next time step
    fn predict_next(&mut self, data_loader: &CSVDataLoader, recent_data: &[DataPoint]) -> Option<f64> {
        if recent_data.len() < self.sequence_length {
            return None;
        }
        
        let trainer = self.trainer.as_mut()?;
        
        let start_idx = recent_data.len() - self.sequence_length;
        let inputs: Vec<Array2<f64>> = recent_data[start_idx..]
            .iter()
            .map(|point| data_loader.normalize(point))
            .collect();
        
        let predictions = trainer.predict(&inputs);
        
        if let Some(prediction) = predictions.last() {
            let normalized_pred = prediction[[0, 0]];
            Some(data_loader.denormalize(normalized_pred, self.target_feature))
        } else {
            None
        }
    }
}

fn main() {
    println!("📈 Real Data Time Series Prediction with LSTM");
    println!("===============================================\n");
    
    // Generate synthetic sensor data (in practice, load from real CSV)
    println!("📡 Generating synthetic IoT sensor data...");
    let mut data_loader = CSVDataLoader::generate_synthetic_sensor_data(7); // 7 days for quick demo
    
    println!("📊 Data loaded: {} data points with {} features",
             data_loader.data.len(),
             data_loader.feature_names.len());
    
    // Display feature names
    println!("Features: {:?}", data_loader.feature_names);
    
    // Show sample data
    println!("\n📋 Sample data points:");
    for (i, point) in data_loader.data.iter().take(5).enumerate() {
        println!("Point {}: {} -> {:?}",
                 i + 1, point.timestamp, 
                 point.values.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>());
    }
    
    // Fit normalizers
    println!("\n🔧 Fitting data normalizers...");
    data_loader.fit_normalizers();
    
    // Create predictor to predict temperature (feature 0)
    let mut predictor = TimeSeriesPredictor::new(
        data_loader.feature_names.len(),  // All features as input
        12,  // 12-hour sequences (reduced for speed)
        32, // 32 hidden units (reduced for speed)
        0,   // Predict temperature (index 0)
    );
    
    // Train the model
    predictor.train(&data_loader, 0.2); // 80% train, 20% validation
    
    // Make predictions on recent data
    println!("\n🔮 Making temperature predictions:");
    let recent_data = &data_loader.data[data_loader.data.len()-48..]; // Last 48 hours
    
    for i in 24..29 { // Predict for hours 25-29
        let input_data = &recent_data[i-24..i];
        if let Some(predicted_temp) = predictor.predict_next(&data_loader, input_data) {
            let actual_temp = recent_data[i].values[0];
            let error = (predicted_temp - actual_temp).abs();
            
            println!("Hour {}: Predicted={:.1}°C, Actual={:.1}°C, Error={:.1}°C",
                     i + 1, predicted_temp, actual_temp, error);
        }
    }
    
    // Calculate statistics
    let temps: Vec<f64> = data_loader.data.iter().map(|p| p.values[0]).collect();
    let avg_temp = temps.iter().sum::<f64>() / temps.len() as f64;
    let temp_range = temps.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &t| {
        (min.min(t), max.max(t))
    });
    
    println!("\n📈 Data statistics:");
    println!("Average temperature: {:.1}°C", avg_temp);
    println!("Temperature range: {:.1}°C to {:.1}°C", temp_range.0, temp_range.1);
} 