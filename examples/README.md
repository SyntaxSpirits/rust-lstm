# 🌍 Real-World LSTM Examples

This directory contains practical examples demonstrating how to use the rust-lstm library for various real-world applications.

## 📊 Examples Overview

### 1. 🏦 **Stock Price Prediction** (`stock_prediction.rs`)
**Use Case**: Financial time series forecasting  
**Features**: OHLCV data processing, z-score normalization, multi-step prediction  
**Demonstrates**: Regression with financial data, risk assessment patterns

```bash
cargo run --example stock_prediction
```

**Key Concepts**:
- Multi-feature financial data (Open, High, Low, Close, Volume)
- Z-score normalization for different value scales
- Sequential pattern recognition in price movements
- Error percentage calculation for trading decisions

**Real-world Extensions**:
- Connect to APIs (Alpha Vantage, Yahoo Finance, Polygon)
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Implement risk management and position sizing
- Include market sentiment and news data

---

### 2. 🌤️ **Weather Forecasting** (`weather_prediction.rs`)
**Use Case**: Meteorological prediction systems  
**Features**: Multi-sensor data fusion, seasonal patterns, correlation modeling  
**Demonstrates**: Environmental time series with correlated features

```bash
cargo run --example weather_prediction
```

**Key Concepts**:
- Multi-variate time series (temperature, humidity, pressure, wind, precipitation)
- Seasonal and daily cycle modeling
- Feature correlation (humidity vs temperature)
- Min-max normalization for different measurement scales

**Real-world Extensions**:
- Integrate with weather station APIs
- Add satellite imagery and radar data
- Implement extreme weather early warning
- Agricultural yield prediction
- Energy demand forecasting

---

### 3. 📝 **Advanced Text Generation** (`text_generation_advanced.rs`)
**Use Case**: Natural language processing and generation  
**Features**: Character-level modeling, vocabulary building, temperature sampling  
**Demonstrates**: Training embedding, LSTM and output layer together by hand with the low-level API

```bash
cargo run --release --example text_generation_advanced
```

**Key Concepts**:
- Character-level language modeling
- `CharacterEmbedding` → `LSTMNetwork` → `LinearLayer` trained end to end with softmax cross-entropy
- Backpropagation through time with `forward_sequence_with_cache` and `backward_sequence`
- Gradients flowing from the loss into the embedding table
- Temperature-controlled sampling from the trained model

**Real-world Extensions**:
- Word-level or subword tokenization (BPE)
- Larger vocabulary and context windows
- Code completion systems
- Chatbot development
- Style transfer applications

---

### 4. 📡 **Real Data Processing** (`real_data_example.rs`)
**Use Case**: IoT sensor monitoring and industrial systems  
**Features**: CSV data loading, automated preprocessing, multi-feature prediction  
**Demonstrates**: Production-ready data pipeline

```bash
cargo run --example real_data_example
```

**Key Concepts**:
- CSV file parsing and error handling
- Automated feature normalization
- Time series sequence creation
- Generic data structures for any domain
- Synthetic data generation for testing

**Real-world Extensions**:
- Database connectivity (PostgreSQL, InfluxDB)
- Real-time streaming data processing
- Anomaly detection systems
- Predictive maintenance
- Quality control in manufacturing

---

### 5. 🎯 **Basic Training** (`training_example.rs`)
**Use Case**: Learning the fundamentals  
**Features**: Sine wave prediction, optimizer comparison  
**Demonstrates**: Core training concepts and validation

```bash
cargo run --example training_example
```

## 🚀 Running Examples

### Prerequisites
```bash
# Ensure you have Rust installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and navigate to the project
git clone https://github.com/SyntaxSpirits/rust-lstm
cd rust-lstm
```

### Run All Examples
```bash
# Financial prediction
cargo run --example stock_prediction

# Weather forecasting  
cargo run --example weather_prediction

# Text generation
cargo run --example text_generation_advanced

# Real data processing
cargo run --example real_data_example

# Basic training concepts
cargo run --example training_example
```

### With Release Optimization
```bash
# For faster execution (especially for text generation)
cargo run --release --example text_generation_advanced
```

## 📈 Expected Outputs

### Stock Prediction
```
🏦 Stock Price Prediction with LSTM
=====================================

📈 Generated 500 days of synthetic stock data
📊 Sample data:
Day 1: Close=$100.12, Volume=950000
...
🔮 Making predictions...
Day 21: Predicted=$102.45, Actual=$102.12, Error=0.3%
```

### Weather Forecasting
```
🌤️ Weather Temperature Prediction with LSTM
===========================================

🌍 Generated 365 days of synthetic weather data
...
🔮 Temperature predictions for next 5 days:
Day 8: Predicted=18.5°C, Actual=18.2°C, Error=0.3°C
```

### Text Generation
```
Character-level LSTM language model
...
Loss of a uniform guess: 2.708

epoch   1: loss 2.7083
epoch  20: loss 0.9542
...
epoch 100: loss 0.1777

temperature 0.2: the dog saw the dog and the dog sat on the log. the cat saw the dog and the dog sat 
temperature 0.5: the dog and the dog sat on the mat. the dog saw the dog and the dog sat on the log. 
temperature 1.0: the cat. the cat saw the cat. the cat saw the cat. the cat saw the dog and the dog s
```

## 🛠️ Customization Guide

### Adding Your Own Data

#### 1. CSV Data Format
```csv
timestamp,feature1,feature2,feature3,target
2024-01-01,1.2,3.4,5.6,7.8
2024-01-02,1.3,3.5,5.7,7.9
```

#### 2. Custom Data Structure
```rust
#[derive(Debug, Clone)]
struct YourDataPoint {
    timestamp: String,
    values: Vec<f64>,
}
```

#### 3. Model Configuration
```rust
// Adjust these parameters for your use case
let sequence_length = 24;  // Look-back window
let hidden_size = 128;     // Model capacity  
let learning_rate = 0.001; // Training speed
let epochs = 100;          // Training iterations
```

### Model Hyperparameters

| Parameter | Small Dataset | Medium Dataset | Large Dataset |
|-----------|--------------|----------------|---------------|
| `hidden_size` | 32-64 | 128-256 | 512-1024 |
| `sequence_length` | 10-20 | 50-100 | 200-500 |
| `learning_rate` | 0.01 | 0.001 | 0.0001 |
| `batch_size` | Full | 32-128 | 256-1024 |

### Loss Functions by Use Case

| Use Case | Loss Function | Output Activation |
|----------|---------------|-------------------|
| Regression | MSELoss | None (linear) |
| Classification | CrossEntropyLoss | Softmax |
| Probability | MAELoss | Sigmoid |

## 🌟 Performance Tips

### Training Speed
- Use `--release` flag for 10-20x speedup
- Larger batch sizes (when implemented)
- GPU acceleration (future feature)

### Memory Optimization
- Shorter sequences for large datasets
- Gradient clipping to prevent exploding gradients
- Smaller hidden sizes for deployment

### Model Quality
- More training epochs with early stopping
- Validation split to prevent overfitting
- Feature engineering and normalization
- Ensemble methods with multiple models

## 🎯 Next Steps

### Production Deployment
1. **Model Serialization**: Save/load trained models
2. **API Integration**: REST/GraphQL endpoints
3. **Database Storage**: Persistent data management
4. **Monitoring**: Performance and drift detection
5. **Scaling**: Distributed training and inference

### Advanced Features
1. **Attention Mechanisms**: Better long-term dependencies
2. **Bidirectional LSTMs**: Past and future context
3. **Multi-task Learning**: Shared representations
4. **Transfer Learning**: Pre-trained models
5. **Hyperparameter Optimization**: Automated tuning

### Domain-Specific Extensions
- **Finance**: Risk models, portfolio optimization
- **Healthcare**: Vital sign monitoring, drug discovery
- **Manufacturing**: Quality control, supply chain
- **Energy**: Grid optimization, renewable forecasting
- **Transportation**: Route optimization, autonomous vehicles

## 📚 Further Reading

- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) - Original LSTM architecture
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Visual explanation
- [Time Series Forecasting](https://otexts.com/fpp3/) - Statistical foundations
- [Deep Learning Book](https://www.deeplearningbook.org/) - Comprehensive theory

## 🤝 Contributing

Have an interesting use case? We'd love to see it!

1. Create a new example file
2. Add it to `Cargo.toml` 
3. Update this README
4. Submit a pull request

**Example domains we're looking for**:
- Computer vision with sequence data
- Audio/speech processing
- Network traffic analysis  
- Social media sentiment
- Bioinformatics sequences
- Game AI and reinforcement learning 