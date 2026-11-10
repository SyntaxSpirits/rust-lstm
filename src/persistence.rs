use ndarray::{Array2, Dimension};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::layers::dropout::{Dropout, Zoneout};
use crate::layers::lstm_cell::LSTMCell;
use crate::models::lstm_network::LSTMNetwork;

const BINARY_MAGIC: &[u8; 8] = b"RLSTMBIN";
const BINARY_FORMAT_VERSION: u32 = 1;

fn invalid(message: String) -> PersistenceError {
    PersistenceError::SerializationError(message)
}

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

impl SerializableArray2 {
    fn into_array(
        self,
        name: &str,
        expected: (usize, usize),
    ) -> Result<Array2<f64>, PersistenceError> {
        if self.shape != expected {
            return Err(invalid(format!(
                "{name} has shape {:?}, expected {:?}",
                self.shape, expected
            )));
        }
        Array2::from_shape_vec(self.shape, self.data)
            .map_err(|err| invalid(format!("{name}: {err}")))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializableDropout {
    dropout_rate: f64,
    variational: bool,
    #[serde(default = "default_training")]
    is_training: bool,
}

impl From<&Dropout> for SerializableDropout {
    fn from(dropout: &Dropout) -> Self {
        Self {
            dropout_rate: dropout.dropout_rate,
            variational: dropout.variational,
            is_training: dropout.is_training,
        }
    }
}

fn valid_rate(name: &str, rate: f64) -> Result<f64, PersistenceError> {
    if (0.0..=1.0).contains(&rate) {
        Ok(rate)
    } else {
        Err(invalid(format!("{name} rate {rate} is outside [0, 1]")))
    }
}

impl SerializableDropout {
    fn into_dropout(self, name: &str) -> Result<Dropout, PersistenceError> {
        let rate = valid_rate(name, self.dropout_rate)?;
        let mut dropout = if self.variational {
            Dropout::variational(rate)
        } else {
            Dropout::new(rate)
        };
        dropout.is_training = self.is_training;
        Ok(dropout)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializableZoneout {
    cell_zoneout_rate: f64,
    hidden_zoneout_rate: f64,
    #[serde(default = "default_training")]
    is_training: bool,
}

fn default_training() -> bool {
    true
}

/// Serializable LSTM cell parameters and dropout configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableLSTMCell {
    w_ih: SerializableArray2,
    w_hh: SerializableArray2,
    b_ih: SerializableArray2,
    b_hh: SerializableArray2,
    hidden_size: usize,
    #[serde(default)]
    input_dropout: Option<SerializableDropout>,
    #[serde(default)]
    recurrent_dropout: Option<SerializableDropout>,
    #[serde(default)]
    output_dropout: Option<SerializableDropout>,
    #[serde(default)]
    cell_update_dropout: Option<SerializableDropout>,
    #[serde(default)]
    zoneout: Option<SerializableZoneout>,
    #[serde(default = "default_training")]
    is_training: bool,
}

impl From<&LSTMCell> for SerializableLSTMCell {
    fn from(cell: &LSTMCell) -> Self {
        Self {
            w_ih: (&cell.w_ih).into(),
            w_hh: (&cell.w_hh).into(),
            b_ih: (&cell.b_ih).into(),
            b_hh: (&cell.b_hh).into(),
            hidden_size: cell.hidden_size,
            input_dropout: cell.input_dropout.as_ref().map(Into::into),
            recurrent_dropout: cell.recurrent_dropout.as_ref().map(Into::into),
            output_dropout: cell.output_dropout.as_ref().map(Into::into),
            cell_update_dropout: cell.cell_update_dropout.as_ref().map(Into::into),
            zoneout: cell.zoneout.as_ref().map(|z| SerializableZoneout {
                cell_zoneout_rate: z.cell_zoneout_rate,
                hidden_zoneout_rate: z.hidden_zoneout_rate,
                is_training: z.is_training,
            }),
            is_training: cell.is_training,
        }
    }
}

impl SerializableLSTMCell {
    fn into_cell(self, input_size: usize) -> Result<LSTMCell, PersistenceError> {
        let hidden = self.hidden_size;
        let gates = hidden
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("hidden size {hidden} is too large")))?;
        let dropout = |d: Option<SerializableDropout>, name: &str| {
            d.map(|d| d.into_dropout(name)).transpose()
        };
        let zoneout = match self.zoneout {
            Some(z) => {
                let mut zoneout = Zoneout::new(
                    valid_rate("cell zoneout", z.cell_zoneout_rate)?,
                    valid_rate("hidden zoneout", z.hidden_zoneout_rate)?,
                );
                zoneout.is_training = z.is_training;
                Some(zoneout)
            }
            None => None,
        };
        Ok(LSTMCell {
            w_ih: self.w_ih.into_array("w_ih", (gates, input_size))?,
            w_hh: self.w_hh.into_array("w_hh", (gates, hidden))?,
            b_ih: self.b_ih.into_array("b_ih", (gates, 1))?,
            b_hh: self.b_hh.into_array("b_hh", (gates, 1))?,
            hidden_size: hidden,
            input_dropout: dropout(self.input_dropout, "input dropout")?,
            recurrent_dropout: dropout(self.recurrent_dropout, "recurrent dropout")?,
            output_dropout: dropout(self.output_dropout, "output dropout")?,
            cell_update_dropout: dropout(self.cell_update_dropout, "cell update dropout")?,
            zoneout,
            is_training: self.is_training,
        })
    }
}

impl TryFrom<SerializableLSTMCell> for LSTMCell {
    type Error = PersistenceError;

    fn try_from(val: SerializableLSTMCell) -> Result<Self, Self::Error> {
        let input_size = val.w_ih.shape.1;
        val.into_cell(input_size)
    }
}

/// Serializable LSTM network
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableLSTMNetwork {
    cells: Vec<SerializableLSTMCell>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    #[serde(default = "default_training")]
    is_training: bool,
}

impl From<&LSTMNetwork> for SerializableLSTMNetwork {
    fn from(network: &LSTMNetwork) -> Self {
        Self {
            cells: network.get_cells().iter().map(|cell| cell.into()).collect(),
            input_size: network.input_size,
            hidden_size: network.hidden_size,
            num_layers: network.num_layers,
            is_training: network.is_training,
        }
    }
}

impl TryFrom<SerializableLSTMNetwork> for LSTMNetwork {
    type Error = PersistenceError;

    fn try_from(val: SerializableLSTMNetwork) -> Result<Self, Self::Error> {
        if val.cells.len() != val.num_layers {
            return Err(invalid(format!(
                "{} cells stored for {} layers",
                val.cells.len(),
                val.num_layers
            )));
        }
        let mut cells = Vec::with_capacity(val.cells.len());
        for (i, cell) in val.cells.into_iter().enumerate() {
            if cell.hidden_size != val.hidden_size {
                return Err(invalid(format!(
                    "layer {i} has hidden size {}, expected {}",
                    cell.hidden_size, val.hidden_size
                )));
            }
            let input_size = if i == 0 {
                val.input_size
            } else {
                val.hidden_size
            };
            cells.push(cell.into_cell(input_size)?);
        }
        let mut network =
            LSTMNetwork::from_cells(cells, val.input_size, val.hidden_size, val.num_layers);
        network.is_training = val.is_training;
        Ok(network)
    }
}

/// Layout written by `save_to_binary` up to version 0.10.0, which stored weights only.
#[derive(Deserialize)]
struct LegacyLSTMCell {
    w_ih: SerializableArray2,
    w_hh: SerializableArray2,
    b_ih: SerializableArray2,
    b_hh: SerializableArray2,
    hidden_size: usize,
}

#[derive(Deserialize)]
struct LegacyLSTMNetwork {
    cells: Vec<LegacyLSTMCell>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
}

#[derive(Deserialize)]
struct LegacySavedModel {
    network: LegacyLSTMNetwork,
    metadata: ModelMetadata,
}

impl From<LegacySavedModel> for SavedModel {
    fn from(legacy: LegacySavedModel) -> Self {
        let network = legacy.network;
        SavedModel {
            network: SerializableLSTMNetwork {
                cells: network
                    .cells
                    .into_iter()
                    .map(|c| SerializableLSTMCell {
                        w_ih: c.w_ih,
                        w_hh: c.w_hh,
                        b_ih: c.b_ih,
                        b_hh: c.b_hh,
                        hidden_size: c.hidden_size,
                        input_dropout: None,
                        recurrent_dropout: None,
                        output_dropout: None,
                        cell_update_dropout: None,
                        zoneout: None,
                        is_training: true,
                    })
                    .collect(),
                input_size: network.input_size,
                hidden_size: network.hidden_size,
                num_layers: network.num_layers,
                is_training: true,
            },
            metadata: legacy.metadata,
        }
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
        file.write_all(BINARY_MAGIC)?;
        file.write_all(&BINARY_FORMAT_VERSION.to_le_bytes())?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Load model from binary format
    pub fn load_from_binary<P: AsRef<Path>>(path: P) -> Result<SavedModel, PersistenceError> {
        let mut file = File::open(path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        let Some(rest) = contents.strip_prefix(BINARY_MAGIC) else {
            let legacy: LegacySavedModel = bincode::deserialize(&contents)?;
            return Ok(legacy.into());
        };
        if rest.len() < 4 {
            return Err(invalid("truncated binary model header".to_string()));
        }
        let (version, body) = rest.split_at(4);
        match u32::from_le_bytes([version[0], version[1], version[2], version[3]]) {
            BINARY_FORMAT_VERSION => Ok(bincode::deserialize(body)?),
            other => Err(invalid(format!("unsupported binary model format {other}"))),
        }
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

        Ok((saved_model.network.try_into()?, saved_model.metadata))
    }
}
