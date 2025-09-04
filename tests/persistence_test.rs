use ndarray::Array2;
use rust_lstm::{
    LSTMNetwork, 
    persistence::{ModelPersistence, PersistentModel, ModelMetadata},
    training::create_basic_trainer,
};
use tempfile::tempdir;

#[test]
fn test_model_metadata_creation() {
    let metadata = ModelMetadata {
        model_name: "test_model".to_string(),
        version: "0.2.0".to_string(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        input_size: 10,
        hidden_size: 20,
        num_layers: 2,
        total_epochs: 100,
        final_loss: Some(0.01),
        description: Some("Test model for validation".to_string()),
    };

    assert_eq!(metadata.model_name, "test_model");
    assert_eq!(metadata.input_size, 10);
    assert_eq!(metadata.hidden_size, 20);
    assert_eq!(metadata.num_layers, 2);
    assert_eq!(metadata.total_epochs, 100);
    assert_eq!(metadata.final_loss, Some(0.01));
}

#[test]
fn test_network_save_load_json() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_model.json");

    // Create a simple network
    let mut network = LSTMNetwork::new(3, 4, 2);
    
    // Test forward pass to ensure network works
    let input = Array2::ones((3, 1));
    let hx = Array2::zeros((4, 1));
    let cx = Array2::zeros((4, 1));
    let (output_before, _) = network.forward(&input, &hx, &cx);

    // Save the network
    let metadata = ModelMetadata {
        model_name: "test_json_model".to_string(),
        version: "0.2.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        input_size: 3,
        hidden_size: 4,
        num_layers: 2,
        total_epochs: 0,
        final_loss: None,
        description: Some("Test JSON persistence".to_string()),
    };

    let result = network.save(&file_path, metadata.clone());
    assert!(result.is_ok());
    assert!(file_path.exists());

    // Load the network
    let (mut loaded_network, loaded_metadata) = LSTMNetwork::load(&file_path).unwrap();
    
    // Verify metadata
    assert_eq!(loaded_metadata.model_name, metadata.model_name);
    assert_eq!(loaded_metadata.input_size, metadata.input_size);
    assert_eq!(loaded_metadata.hidden_size, metadata.hidden_size);
    assert_eq!(loaded_metadata.num_layers, metadata.num_layers);

    // Verify network structure
    assert_eq!(loaded_network.input_size, 3);
    assert_eq!(loaded_network.hidden_size, 4);
    assert_eq!(loaded_network.num_layers, 2);

    // Test that loaded network produces same output (within numerical tolerance)
    let (output_after, _) = loaded_network.forward(&input, &hx, &cx);
    assert_eq!(output_before.shape(), output_after.shape());
    
    // Check if outputs are approximately equal (they should be identical for same weights)
    let diff = (&output_before - &output_after).mapv(|x| x.abs()).sum();
    assert!(diff < 1e-10, "Loaded network output differs significantly from original");
}

#[test]
fn test_network_save_load_binary() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_model.bin");

    // Create a network with dropout
    let mut network = LSTMNetwork::new(2, 3, 1)
        .with_input_dropout(0.1, false)
        .with_output_dropout(0.1);
    
    // Test forward pass
    let input = Array2::ones((2, 1));
    let hx = Array2::zeros((3, 1));
    let cx = Array2::zeros((3, 1));
    network.eval(); // Set to eval mode for consistent output
    let (output_before, _) = network.forward(&input, &hx, &cx);

    // Save the network
    let metadata = ModelMetadata {
        model_name: "test_binary_model".to_string(),
        version: "0.2.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        input_size: 2,
        hidden_size: 3,
        num_layers: 1,
        total_epochs: 50,
        final_loss: Some(0.05),
        description: Some("Test binary persistence".to_string()),
    };

    let result = network.save(&file_path, metadata.clone());
    assert!(result.is_ok());
    assert!(file_path.exists());

    // Load the network
    let (mut loaded_network, loaded_metadata) = LSTMNetwork::load(&file_path).unwrap();
    
    // Verify metadata
    assert_eq!(loaded_metadata.model_name, metadata.model_name);
    assert_eq!(loaded_metadata.total_epochs, metadata.total_epochs);
    assert_eq!(loaded_metadata.final_loss, metadata.final_loss);

    // Verify network structure
    assert_eq!(loaded_network.input_size, 2);
    assert_eq!(loaded_network.hidden_size, 3);
    assert_eq!(loaded_network.num_layers, 1);

    // Test that loaded network produces same output
    loaded_network.eval();
    let (output_after, _) = loaded_network.forward(&input, &hx, &cx);
    assert_eq!(output_before.shape(), output_after.shape());
    
    let diff = (&output_before - &output_after).mapv(|x| x.abs()).sum();
    assert!(diff < 1e-10, "Loaded network output differs significantly from original");
}

#[test]
fn test_model_persistence_create_saved_model() {
    let network = LSTMNetwork::new(5, 10, 3);
    
    let saved_model = ModelPersistence::create_saved_model(
        &network,
        "test_create_model".to_string(),
        200,
        Some(0.001),
        Some("Created via ModelPersistence".to_string()),
    );

    assert_eq!(saved_model.metadata.model_name, "test_create_model");
    assert_eq!(saved_model.metadata.total_epochs, 200);
    assert_eq!(saved_model.metadata.final_loss, Some(0.001));
    assert_eq!(saved_model.metadata.input_size, 5);
    assert_eq!(saved_model.metadata.hidden_size, 10);
    assert_eq!(saved_model.metadata.num_layers, 3);
}

#[test]
fn test_persistence_with_trained_model() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("trained_model.json");

    // Create and train a small model
    let network = LSTMNetwork::new(1, 2, 1);
    let mut trainer = create_basic_trainer(network, 0.01);

    // Generate minimal training data
    let train_data = vec![
        (vec![Array2::ones((1, 1))], vec![Array2::zeros((2, 1))]),
        (vec![Array2::zeros((1, 1))], vec![Array2::ones((2, 1))]),
    ];

    // Quick training for 2 epochs
    let config = rust_lstm::training::TrainingConfig {
        epochs: 2,
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None,
    };
    trainer = trainer.with_config(config);
    trainer.train(&train_data, None);

    let final_loss = trainer.get_latest_metrics().map(|m| m.train_loss);

    // Save the trained model
    let metadata = ModelMetadata {
        model_name: "trained_test_model".to_string(),
        version: "0.2.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        input_size: 1,
        hidden_size: 2,
        num_layers: 1,
        total_epochs: 2,
        final_loss,
        description: Some("Trained model test".to_string()),
    };

    let save_result = trainer.network.save(&file_path, metadata);
    assert!(save_result.is_ok());

    // Load and verify
    let (mut loaded_network, loaded_metadata) = LSTMNetwork::load(&file_path).unwrap();
    assert_eq!(loaded_metadata.model_name, "trained_test_model");
    assert_eq!(loaded_metadata.total_epochs, 2);
    
    // Test that loaded model can make predictions
    let test_input = vec![Array2::ones((1, 1))];
    loaded_network.eval();
    
    let hx = Array2::zeros((2, 1));
    let cx = Array2::zeros((2, 1));
    let (output, _) = loaded_network.forward(&test_input[0], &hx, &cx);
    
    assert_eq!(output.shape(), &[2, 1]);
}

#[test]
fn test_file_extension_detection() {
    let dir = tempdir().unwrap();
    let network = LSTMNetwork::new(2, 3, 1);
    
    let metadata = ModelMetadata {
        model_name: "extension_test".to_string(),
        version: "0.2.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        input_size: 2,
        hidden_size: 3,
        num_layers: 1,
        total_epochs: 0,
        final_loss: None,
        description: None,
    };

    // Test JSON extension
    let json_path = dir.path().join("model.json");
    let result = network.save(&json_path, metadata.clone());
    assert!(result.is_ok());
    assert!(json_path.exists());

    // Test binary extension
    let bin_path = dir.path().join("model.bin");
    let result = network.save(&bin_path, metadata.clone());
    assert!(result.is_ok());
    assert!(bin_path.exists());

    // Test .model extension (should default to binary)
    let model_path = dir.path().join("model.model");
    let result = network.save(&model_path, metadata.clone());
    assert!(result.is_ok());
    assert!(model_path.exists());

    // Test unknown extension (should default to binary)
    let unknown_path = dir.path().join("model.xyz");
    let result = network.save(&unknown_path, metadata);
    assert!(result.is_ok());
    assert!(unknown_path.exists());
}

#[test]
fn test_error_handling() {
    // Test loading non-existent file
    let result = LSTMNetwork::load("/non/existent/path.json");
    assert!(result.is_err());

    // Test saving to invalid path (should fail gracefully)
    let network = LSTMNetwork::new(1, 1, 1);
    let metadata = ModelMetadata {
        model_name: "error_test".to_string(),
        version: "0.2.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        input_size: 1,
        hidden_size: 1,
        num_layers: 1,
        total_epochs: 0,
        final_loss: None,
        description: None,
    };

    let result = network.save("/invalid/path/that/does/not/exist.json", metadata);
    assert!(result.is_err());
} 