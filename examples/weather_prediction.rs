use ndarray::{Array2, arr2};
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::MSELoss;
use rust_lstm::optimizers::Adam;

/// Weather data with multiple meteorological features
#[derive(Debug, Clone)]
struct WeatherData {
    date: String,
    temperature: f64,    // °C
    humidity: f64,       // %
    pressure: f64,       // hPa
    wind_speed: f64,     // km/h
    precipitation: f64,  // mm
    cloud_cover: f64,    // %
}

/// Multi-feature weather prediction system
struct WeatherPredictor {
    network: LSTMNetwork,
    trainer: Option<LSTMTrainer<MSELoss, Adam>>,
    feature_scalers: Vec<(f64, f64)>, // (min, max) for each feature
    sequence_length: usize,
}

impl WeatherPredictor {
    fn new(sequence_length: usize, hidden_size: usize) -> Self {
        // 6 input features, predicting temperature
        let network = LSTMNetwork::new(6, hidden_size, 2);
        
        Self {
            network,
            trainer: None,
            feature_scalers: vec![(0.0, 1.0); 6],
            sequence_length,
        }
    }

    /// Fit min-max scalers for normalization
    fn fit_scalers(&mut self, data: &[WeatherData]) {
        let mut mins = vec![f64::INFINITY; 6];
        let mut maxs = vec![f64::NEG_INFINITY; 6];

        for weather in data {
            let features = self.extract_features(weather);
            for (i, &value) in features.iter().enumerate() {
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
        }

        for i in 0..6 {
            // Add small margin to avoid edge cases
            let range = maxs[i] - mins[i];
            mins[i] -= range * 0.01;
            maxs[i] += range * 0.01;
            self.feature_scalers[i] = (mins[i], maxs[i]);
        }
    }

    /// Extract numerical features from weather data
    fn extract_features(&self, weather: &WeatherData) -> Vec<f64> {
        vec![
            weather.temperature,
            weather.humidity,
            weather.pressure,
            weather.wind_speed,
            weather.precipitation,
            weather.cloud_cover,
        ]
    }

    /// Normalize features to [0, 1] range
    fn normalize_features(&self, weather: &WeatherData) -> Array2<f64> {
        let features = self.extract_features(weather);
        let normalized: Vec<f64> = features.iter().enumerate()
            .map(|(i, &value)| {
                let (min_val, max_val) = self.feature_scalers[i];
                (value - min_val) / (max_val - min_val)
            })
            .collect();

        Array2::from_shape_vec((6, 1), normalized).unwrap()
    }

    /// Denormalize temperature prediction
    fn denormalize_temperature(&self, normalized_temp: f64) -> f64 {
        let (min_temp, max_temp) = self.feature_scalers[0]; // Temperature is first feature
        normalized_temp * (max_temp - min_temp) + min_temp
    }

    /// Create training sequences for weather prediction
    fn create_sequences(&self, data: &[WeatherData]) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
        let mut sequences = Vec::new();

        for i in 0..data.len().saturating_sub(self.sequence_length) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();

            // Input sequence: past weather conditions
            for j in i..i + self.sequence_length {
                inputs.push(self.normalize_features(&data[j]));
            }

            // Target sequence: predict next day's temperature at each timestep
            for j in i + 1..i + self.sequence_length + 1 {
                if j < data.len() {
                    let next_temp = data[j].temperature;
                    let (min_temp, max_temp) = self.feature_scalers[0];
                    let normalized_target = (next_temp - min_temp) / (max_temp - min_temp);
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

    /// Train the weather prediction model
    fn train(&mut self, data: &[WeatherData], validation_split: f64) {
        println!("🌡️ Fitting scalers on {} data points...", data.len());
        self.fit_scalers(data);

        println!("🔄 Creating training sequences...");
        let sequences = self.create_sequences(data);

        let split_idx = ((sequences.len() as f64) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = sequences.split_at(split_idx);

        println!("🎯 Training on {} sequences, validating on {} sequences",
                train_data.len(), val_data.len());

        let loss_function = MSELoss;
        let optimizer = Adam::new(0.001);
        let mut trainer = LSTMTrainer::new(self.network.clone(), loss_function, optimizer);

        // Configure training for quicker demo
        let mut config = rust_lstm::training::TrainingConfig::default();
        config.epochs = 20; // Reduced from 100 for demo
        config.print_every = 5; // Print more frequently
        trainer = trainer.with_config(config);

        trainer.train(train_data, Some(val_data));

        self.trainer = Some(trainer);
        println!("✅ Weather model training completed!");
    }

    /// Predict next day's temperature
    fn predict_temperature(&self, recent_data: &[WeatherData]) -> Option<f64> {
        if recent_data.len() < self.sequence_length {
            return None;
        }

        let trainer = self.trainer.as_ref()?;

        let start_idx = recent_data.len() - self.sequence_length;
        let inputs: Vec<Array2<f64>> = recent_data[start_idx..]
            .iter()
            .map(|weather| self.normalize_features(weather))
            .collect();

        let predictions = trainer.predict(&inputs);

        if let Some(prediction) = predictions.last() {
            let normalized_temp = prediction[[0, 0]];
            Some(self.denormalize_temperature(normalized_temp))
        } else {
            None
        }
    }
}

/// Generate realistic weather data with seasonal patterns
fn generate_weather_data(days: usize) -> Vec<WeatherData> {
    let mut data = Vec::new();

    for i in 0..days {
        let day_of_year = (i % 365) as f64;
        
        // Seasonal temperature variation
        let seasonal_temp = 15.0 + 10.0 * (2.0 * std::f64::consts::PI * day_of_year / 365.0).sin();
        
        // Daily temperature variation with some randomness
        let daily_variation = (rand::random::<f64>() - 0.5) * 6.0;
        let temperature = seasonal_temp + daily_variation;
        
        // Humidity inversely correlated with temperature
        let humidity = 70.0 - (temperature - 15.0) * 2.0 + (rand::random::<f64>() - 0.5) * 20.0;
        let humidity = humidity.clamp(20.0, 95.0);
        
        // Pressure with weather patterns
        let pressure = 1013.25 + (rand::random::<f64>() - 0.5) * 30.0;
        
        // Wind speed with some correlation to pressure changes
        let wind_speed = 10.0 + (rand::random::<f64>() * 15.0);
        
        // Precipitation probability based on humidity and pressure
        let precip_prob = (humidity - 50.0) / 100.0 + (1020.0 - pressure) / 50.0;
        let precipitation = if rand::random::<f64>() < precip_prob.max(0.0) {
            rand::random::<f64>() * 15.0 // 0-15mm
        } else {
            0.0
        };
        
        // Cloud cover correlated with precipitation and humidity
        let cloud_cover = (humidity - 30.0) / 70.0 * 100.0 + 
                         if precipitation > 0.0 { 30.0 } else { 0.0 };
        let cloud_cover = cloud_cover.clamp(0.0, 100.0);

        data.push(WeatherData {
            date: format!("2024-{:03}", i + 1),
            temperature,
            humidity,
            pressure,
            wind_speed,
            precipitation,
            cloud_cover,
        });
    }

    data
}

fn main() {
    println!("🌤️ Weather Temperature Prediction with LSTM");
    println!("===========================================\n");

    // Generate synthetic weather data
    let weather_data = generate_weather_data(365); // One year of data
    println!("🌍 Generated {} days of synthetic weather data", weather_data.len());

    // Print sample data
    println!("\n📊 Sample weather data:");
    for (i, weather) in weather_data.iter().take(5).enumerate() {
        println!("Day {}: Temp={:.1}°C, Humidity={:.0}%, Pressure={:.1}hPa, Precip={:.1}mm",
                i + 1, weather.temperature, weather.humidity, weather.pressure, weather.precipitation);
    }

    // Create and train predictor
    let mut predictor = WeatherPredictor::new(7, 64); // 7-day sequences, 64 hidden units
    predictor.train(&weather_data, 0.2); // 80% train, 20% validation

    // Make temperature predictions
    println!("\n🔮 Temperature predictions for next 5 days:");
    let recent_data = &weather_data[weather_data.len()-20..]; // Last 20 days

    for i in 7..12 { // Predict for days 8-12 of recent data
        let input_data = &recent_data[i-7..i];
        if let Some(predicted_temp) = predictor.predict_temperature(input_data) {
            let actual_temp = recent_data[i].temperature;
            let error = (predicted_temp - actual_temp).abs();
            
            println!("Day {}: Predicted={:.1}°C, Actual={:.1}°C, Error={:.1}°C",
                    i + 1, predicted_temp, actual_temp, error);
        }
    }

    // Calculate seasonal statistics
    let temps: Vec<f64> = weather_data.iter().map(|w| w.temperature).collect();
    let avg_temp = temps.iter().sum::<f64>() / temps.len() as f64;
    let min_temp = temps.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_temp = temps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n📈 Annual temperature statistics:");
    println!("Average: {:.1}°C", avg_temp);
    println!("Range: {:.1}°C to {:.1}°C", min_temp, max_temp);
} 