#![allow(clippy::type_complexity)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use ndarray::Array2;
use rust_lstm::{
    persistence::{ModelMetadata, ModelPersistence, PersistenceError, PersistentModel},
    training::create_basic_trainer,
    LSTMNetwork,
};
use std::path::Path;
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
    let state = network.zero_state(1);
    let (output_before, _) = network.forward(&input, &state);

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
    let (output_after, _) = loaded_network.forward(&input, &state);
    assert_eq!(output_before.shape(), output_after.shape());

    // Check if outputs are approximately equal (they should be identical for same weights)
    let diff = (&output_before - &output_after).mapv(|x| x.abs()).sum();
    assert!(
        diff < 1e-10,
        "Loaded network output differs significantly from original"
    );
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
    let state = network.zero_state(1);
    network.eval(); // Set to eval mode for consistent output
    let (output_before, _) = network.forward(&input, &state);

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
    let (output_after, _) = loaded_network.forward(&input, &state);
    assert_eq!(output_before.shape(), output_after.shape());

    let diff = (&output_before - &output_after).mapv(|x| x.abs()).sum();
    assert!(
        diff < 1e-10,
        "Loaded network output differs significantly from original"
    );
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
    let test_input = [Array2::ones((1, 1))];
    loaded_network.eval();

    let state = loaded_network.zero_state(1);
    let (output, _) = loaded_network.forward(&test_input[0], &state);

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

fn fixture_metadata(name: &str) -> ModelMetadata {
    ModelMetadata {
        model_name: name.to_string(),
        version: "0.10.0".to_string(),
        created_at: "2026-10-07T00:00:00+00:00".to_string(),
        input_size: 3,
        hidden_size: 4,
        num_layers: 2,
        total_epochs: 0,
        final_loss: None,
        description: None,
    }
}

fn fixture_inputs() -> Vec<Array2<f64>> {
    (0..3)
        .map(|t| Array2::from_shape_fn((3, 1), |(i, _)| ((t * 3 + i) as f64 * 0.37).sin()))
        .collect()
}

fn cell_config(cell: &rust_lstm::LSTMCell) -> Vec<Option<(f64, bool, bool)>> {
    let mut config: Vec<_> = [
        &cell.input_dropout,
        &cell.recurrent_dropout,
        &cell.output_dropout,
        &cell.cell_update_dropout,
    ]
    .iter()
    .map(|d| {
        d.as_ref()
            .map(|d| (d.dropout_rate, d.variational, d.is_training))
    })
    .collect();
    config.push(cell.zoneout.as_ref().map(|z| {
        (
            z.cell_zoneout_rate + 10.0 * z.hidden_zoneout_rate,
            false,
            z.is_training,
        )
    }));
    config
}

#[test]
fn save_load_keeps_dropout_zoneout_and_mode() {
    let dir = tempdir().unwrap();
    let mut network = LSTMNetwork::new(3, 4, 2)
        .with_input_dropout(0.2, true)
        .with_recurrent_dropout(0.3, false)
        .with_output_dropout(0.25)
        .with_cell_update_dropout(0.15, true)
        .with_zoneout(0.4, 0.6);
    network.eval();
    let xs = fixture_inputs();
    let expected = network.forward_sequence(&xs);

    for file in ["config.json", "config.bin"] {
        let path = dir.path().join(file);
        network.save(&path, fixture_metadata(file)).unwrap();
        let (mut loaded, _) = LSTMNetwork::load(&path).unwrap();

        assert!(!loaded.is_training, "{file}");
        for (a, b) in network.get_cells().iter().zip(loaded.get_cells()) {
            assert!(!b.is_training, "{file}");
            assert_eq!(cell_config(a), cell_config(b), "{file}");
        }
        let outputs = loaded.forward_sequence(&xs);
        for (a, b) in expected.iter().zip(&outputs) {
            assert!((a - b).iter().all(|d| d.abs() < 1e-15), "{file}");
        }

        let mut without_zoneout = loaded.clone();
        for cell in without_zoneout.get_cells_mut() {
            cell.zoneout = None;
        }
        let stripped = without_zoneout.forward_sequence(&xs);
        assert!((&stripped[2] - &outputs[2]).iter().any(|d| d.abs() > 1e-6));
    }

    network.train();
    let path = dir.path().join("training.json");
    network.save(&path, fixture_metadata("training")).unwrap();
    let (loaded, _) = LSTMNetwork::load(&path).unwrap();
    assert!(loaded.is_training);
    assert!(loaded.get_cells().iter().all(|c| c.is_training));
}

#[test]
fn save_load_keeps_component_modes() {
    let dir = tempdir().unwrap();
    let mut network = LSTMNetwork::new(3, 4, 2)
        .with_input_dropout(0.2, true)
        .with_recurrent_dropout(0.3, false)
        .with_zoneout(0.4, 0.6);
    network.get_cells_mut()[0].zoneout.as_mut().unwrap().eval();
    network.get_cells_mut()[1]
        .recurrent_dropout
        .as_mut()
        .unwrap()
        .eval();
    network.eval();
    network.get_cells_mut()[1]
        .input_dropout
        .as_mut()
        .unwrap()
        .train();

    for file in ["modes.json", "modes.bin"] {
        let path = dir.path().join(file);
        network.save(&path, fixture_metadata(file)).unwrap();
        let (loaded, _) = LSTMNetwork::load(&path).unwrap();
        for (a, b) in network.get_cells().iter().zip(loaded.get_cells()) {
            assert_eq!(a.is_training, b.is_training, "{file}");
            assert_eq!(cell_config(a), cell_config(b), "{file}");
        }
    }

    let mut training = network.clone();
    training.train();
    training.get_cells_mut()[0].zoneout.as_mut().unwrap().eval();
    let path = dir.path().join("zoneout_eval.bin");
    training
        .save(&path, fixture_metadata("zoneout_eval"))
        .unwrap();
    let (loaded, _) = LSTMNetwork::load(&path).unwrap();
    let cell = &loaded.get_cells()[0];
    assert!(cell.is_training);
    assert!(!cell.zoneout.as_ref().unwrap().is_training);
    assert!(cell.input_dropout.as_ref().unwrap().is_training);
}

/// Files under tests/fixtures/v0_10_0 were written by `PersistentModel::save` of
/// rust-lstm 0.10.0, together with the output of the saved network on `fixture_inputs`.
#[test]
fn loads_models_saved_by_0_10_0() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0_10_0");
    let expected: Vec<f64> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("lstm_3_4_2_output.json")).unwrap())
            .unwrap();

    let mut weights = Vec::new();
    for file in ["lstm_3_4_2.json", "lstm_3_4_2.bin"] {
        let (mut network, metadata) = LSTMNetwork::load(dir.join(file)).unwrap();
        assert_eq!(metadata.model_name, "lstm_3_4_2", "{file}");
        assert_eq!(metadata.version, "0.10.0", "{file}");
        assert_eq!(metadata.total_epochs, 5, "{file}");
        assert_eq!(metadata.final_loss, Some(0.125), "{file}");
        assert_eq!(
            (network.input_size, network.hidden_size, network.num_layers),
            (3, 4, 2)
        );
        assert!(network.is_training);
        for cell in network.get_cells() {
            assert!(cell.is_training);
            assert!(cell_config(cell).iter().all(Option::is_none));
        }

        let output = network.forward_sequence(&fixture_inputs()).pop().unwrap();
        assert_eq!(output.len(), expected.len());
        for (a, b) in output.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-15, "{file}: {a} vs {b}");
        }
        weights.push(
            network
                .get_cells()
                .iter()
                .flat_map(|c| [&c.w_ih, &c.w_hh, &c.b_ih, &c.b_hh])
                .flat_map(|m| m.iter().copied().collect::<Vec<_>>())
                .collect::<Vec<f64>>(),
        );
    }
    assert_eq!(weights[0].len(), weights[1].len());
    assert!(weights[0]
        .iter()
        .zip(&weights[1])
        .all(|(a, b)| (a - b).abs() < 1e-15));
}

fn saved_json(dir: &Path) -> serde_json::Value {
    let path = dir.join("valid.json");
    LSTMNetwork::new(3, 4, 2)
        .with_input_dropout(0.1, false)
        .save(&path, fixture_metadata("valid"))
        .unwrap();
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn assert_load_fails(path: &Path) {
    match LSTMNetwork::load(path) {
        Err(PersistenceError::SerializationError(_)) => {}
        Err(other) => panic!("unexpected error {other}"),
        Ok(_) => panic!("{} loaded", path.display()),
    }
}

#[test]
fn malformed_files_return_errors() {
    let dir = tempdir().unwrap();
    let corruptions: [(&str, fn(&mut serde_json::Value)); 5] = [
        ("short_data", |v| {
            v["network"]["cells"][0]["w_ih"]["data"]
                .as_array_mut()
                .unwrap()
                .pop();
        }),
        ("transposed", |v| {
            v["network"]["cells"][1]["w_hh"]["shape"] = serde_json::json!([4, 16]);
        }),
        ("wrong_input_size", |v| {
            v["network"]["input_size"] = serde_json::json!(5);
        }),
        ("missing_layer", |v| {
            v["network"]["cells"].as_array_mut().unwrap().pop();
        }),
        ("dropout_rate", |v| {
            v["network"]["cells"][0]["input_dropout"]["dropout_rate"] = serde_json::json!(1.5);
        }),
    ];
    for (name, corrupt) in corruptions {
        let mut value = saved_json(dir.path());
        corrupt(&mut value);
        let path = dir.path().join(format!("{name}.json"));
        std::fs::write(&path, value.to_string()).unwrap();
        assert_load_fails(&path);
    }

    let valid = dir.path().join("valid.bin");
    LSTMNetwork::new(3, 4, 2)
        .save(&valid, fixture_metadata("valid"))
        .unwrap();
    let bytes = std::fs::read(&valid).unwrap();
    let mut future = bytes[..8].to_vec();
    future.extend_from_slice(&99u32.to_le_bytes());
    future.extend_from_slice(&bytes[12..]);
    let legacy = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0_10_0/lstm_3_4_2.bin"),
    )
    .unwrap();
    for (name, contents) in [
        ("truncated", bytes[..bytes.len() / 2].to_vec()),
        ("header_only", bytes[..10].to_vec()),
        ("future_version", future),
        ("legacy_truncated", legacy[..legacy.len() - 40].to_vec()),
    ] {
        let path = dir.path().join(format!("{name}.bin"));
        std::fs::write(&path, contents).unwrap();
        assert_load_fails(&path);
    }
}
