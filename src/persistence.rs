use ndarray::{Array2, Dimension};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::layers::lstm_cell::LSTMCell;
use crate::models::lstm_network::LSTMNetwork;

/// Serializable version of Array2<f64> for persistence
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializableArray2 {
    data: Vec<f64>,
    shape: (usize, usize),
}

impl From<&Array2<f64>> for SerializableArray2 {
    fn from(array: &Array2<f64>) -> Self {
        Self {
            data: array.iter().cloned().collect(),
            shape: array.raw_dim().into_pattern(),
        }
    }
}

impl From<SerializableArray2> for Array2<f64> {
    fn from(val: SerializableArray2) -> Self {
        Array2::from_shape_vec(val.shape, val.data)
            .expect("Failed to reconstruct Array2 from serialized data")
    }
}

/// Serializable LSTM cell parameters
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableLSTMCell {
    w_ih: SerializableArray2,
    w_hh: SerializableArray2,
    b_ih: SerializableArray2,
    b_hh: SerializableArray2,
    hidden_size: usize,
}

impl From<&LSTMCell> for SerializableLSTMCell {
    fn from(cell: &LSTMCell) -> Self {
        Self {
            w_ih: (&cell.w_ih).into(),
            w_hh: (&cell.w_hh).into(),
            b_ih: (&cell.b_ih).into(),
            b_hh: (&cell.b_hh).into(),
            hidden_size: cell.hidden_size,
        }
    }
}

impl From<SerializableLSTMCell> for LSTMCell {
    fn from(val: SerializableLSTMCell) -> Self {
        LSTMCell {
            w_ih: val.w_ih.into(),
            w_hh: val.w_hh.into(),
            b_ih: val.b_ih.into(),
            b_hh: val.b_hh.into(),
            hidden_size: val.hidden_size,
            input_dropout: None,
            recurrent_dropout: None,
            output_dropout: None,
            zoneout: None,
            is_training: true,
        }
    }
}

/// Serializable LSTM network
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableLSTMNetwork {
    cells: Vec<SerializableLSTMCell>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
}

impl From<&LSTMNetwork> for SerializableLSTMNetwork {
    fn from(network: &LSTMNetwork) -> Self {
        Self {
            cells: network.get_cells().iter().map(|cell| cell.into()).collect(),
            input_size: network.input_size,
            hidden_size: network.hidden_size,
            num_layers: network.num_layers,
        }
    }
}

impl From<SerializableLSTMNetwork> for LSTMNetwork {
    fn from(val: SerializableLSTMNetwork) -> Self {
        LSTMNetwork::from_cells(
            val.cells.into_iter().map(|cell| cell.into()).collect(),
            val.input_size,
            val.hidden_size,
            val.num_layers,
        )
    }
}

/// Model metadata for tracking training information
#[derive(Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub model_name: String,
    pub version: String,
    pub created_at: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub total_epochs: usize,
    pub final_loss: Option<f64>,
    pub description: Option<String>,
}

/// Complete saved model including network and metadata
#[derive(Serialize, Deserialize)]
pub struct SavedModel {
    pub network: SerializableLSTMNetwork,
    pub metadata: ModelMetadata,
}

/// Errors that can occur during model persistence operations
#[derive(Debug)]
pub enum PersistenceError {
    IoError(std::io::Error),
    SerializationError(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::IoError(err) => write!(f, "IO error: {}", err),
            PersistenceError::SerializationError(err) => write!(f, "Serialization error: {}", err),
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PersistenceError::IoError(err) => Some(err),
            PersistenceError::SerializationError(_) => None,
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(error: std::io::Error) -> Self {
        PersistenceError::IoError(error)
    }
}

impl From<serde_json::Error> for PersistenceError {
    fn from(error: serde_json::Error) -> Self {
        PersistenceError::SerializationError(error.to_string())
    }
}

impl From<bincode::Error> for PersistenceError {
    fn from(error: bincode::Error) -> Self {
        PersistenceError::SerializationError(error.to_string())
    }
}

/// Model persistence operations
pub struct ModelPersistence;

impl ModelPersistence {
    /// Save model to JSON format (human-readable)
    pub fn save_to_json<P: AsRef<Path>>(
        model: &SavedModel,
        path: P,
    ) -> Result<(), PersistenceError> {
        let json = serde_json::to_string_pretty(model)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load model from JSON format
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<SavedModel, PersistenceError> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model = serde_json::from_str(&contents)?;
        Ok(model)
    }

    /// Save model to binary format (compact and fast)
    pub fn save_to_binary<P: AsRef<Path>>(
        model: &SavedModel,
        path: P,
    ) -> Result<(), PersistenceError> {
        let encoded = bincode::serialize(model)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Load model from binary format
    pub fn load_from_binary<P: AsRef<Path>>(path: P) -> Result<SavedModel, PersistenceError> {
        let mut file = File::open(path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        let model = bincode::deserialize(&contents)?;
        Ok(model)
    }

    /// Create a model with metadata
    pub fn create_saved_model(
        network: &LSTMNetwork,
        model_name: String,
        total_epochs: usize,
        final_loss: Option<f64>,
        description: Option<String>,
    ) -> SavedModel {
        let metadata = ModelMetadata {
            model_name,
            version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            input_size: network.input_size,
            hidden_size: network.hidden_size,
            num_layers: network.num_layers,
            total_epochs,
            final_loss,
            description,
        };

        SavedModel {
            network: network.into(),
            metadata,
        }
    }
}

/// Convenience trait for easy model saving/loading
pub trait PersistentModel {
    /// Save model to file (format determined by file extension)
    fn save<P: AsRef<Path>>(
        &self,
        path: P,
        metadata: ModelMetadata,
    ) -> Result<(), PersistenceError>;

    /// Load model from file (format determined by file extension)
    fn load<P: AsRef<Path>>(path: P) -> Result<(Self, ModelMetadata), PersistenceError>
    where
        Self: Sized;
}

impl PersistentModel for LSTMNetwork {
    fn save<P: AsRef<Path>>(
        &self,
        path: P,
        metadata: ModelMetadata,
    ) -> Result<(), PersistenceError> {
        let saved_model = SavedModel {
            network: self.into(),
            metadata,
        };

        let path_ref = path.as_ref();
        match path_ref.extension().and_then(|s| s.to_str()) {
            Some("json") => ModelPersistence::save_to_json(&saved_model, path),
            Some("bin") | Some("model") => ModelPersistence::save_to_binary(&saved_model, path),
            _ => ModelPersistence::save_to_binary(&saved_model, path), // Default to binary
        }
    }

    fn load<P: AsRef<Path>>(path: P) -> Result<(Self, ModelMetadata), PersistenceError> {
        let path_ref = path.as_ref();
        let saved_model = match path_ref.extension().and_then(|s| s.to_str()) {
            Some("json") => ModelPersistence::load_from_json(path)?,
            Some("bin") | Some("model") => ModelPersistence::load_from_binary(path)?,
            _ => ModelPersistence::load_from_binary(path)?, // Default to binary
        };

        Ok((saved_model.network.into(), saved_model.metadata))
    }
}
