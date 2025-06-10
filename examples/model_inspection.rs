use rust_lstm::{LSTMNetwork, ModelMetadata, PersistentModel};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 LSTM Model Internal Structure Inspection");
    println!("============================================\n");

    // Create a simple model for inspection
    println!("🏗️ Creating a simple LSTM model...");
    let input_size = 3;   // e.g., word embeddings of size 3
    let hidden_size = 4;  // 4 hidden units  
    let num_layers = 2;   // 2-layer network
    
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);
    
    // Save it to examine the structure
    let metadata = ModelMetadata {
        model_name: "Language Model Example".to_string(),
        version: "1.0.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        input_size,
        hidden_size,
        num_layers,
        total_epochs: 0,
        final_loss: None,
        description: Some("Example LSTM for language modeling inspection".to_string()),
    };
    
    std::fs::create_dir_all("models")?;
    network.save("models/inspection_model.json", metadata.clone())?;
    
    // Now inspect what's inside
    println!("📊 WHAT'S STORED IN AN LSTM MODEL:");
    println!("=================================\n");
    
    inspect_architecture(&network);
    inspect_parameters(&network);
    inspect_file_contents()?;
    explain_llm_context();
    calculate_parameter_count(&network);
    
    Ok(())
}

fn inspect_architecture(network: &LSTMNetwork) {
    println!("🏗️ NETWORK ARCHITECTURE:");
    println!("  📐 Input Size: {} (size of input vectors, e.g., word embeddings)", network.input_size);
    println!("  🧠 Hidden Size: {} (memory capacity per layer)", network.hidden_size);
    println!("  📚 Number of Layers: {} (depth of the network)", network.num_layers);
    println!("  🔄 Total Cells: {} LSTM cells", network.num_layers);
    println!();
}

fn inspect_parameters(network: &LSTMNetwork) {
    println!("⚙️ PARAMETERS STORED FOR EACH LSTM CELL:");
    println!("==========================================");
    
    for (layer_idx, cell) in network.get_cells().iter().enumerate() {
        println!("📦 Layer {} LSTM Cell:", layer_idx);
        
        // Each LSTM cell contains 4 gates: input, forget, cell, output
        println!("  🔢 w_ih (Input→Hidden weights): {}×{} = {} parameters", 
                 cell.w_ih.shape()[0], cell.w_ih.shape()[1], 
                 cell.w_ih.shape()[0] * cell.w_ih.shape()[1]);
        println!("     • Controls how input affects all 4 gates (i,f,g,o)");
        
        println!("  🔄 w_hh (Hidden→Hidden weights): {}×{} = {} parameters", 
                 cell.w_hh.shape()[0], cell.w_hh.shape()[1],
                 cell.w_hh.shape()[0] * cell.w_hh.shape()[1]);
        println!("     • Controls how previous hidden state affects gates");
        
        println!("  ➕ b_ih (Input biases): {}×{} = {} parameters", 
                 cell.b_ih.shape()[0], cell.b_ih.shape()[1],
                 cell.b_ih.shape()[0] * cell.b_ih.shape()[1]);
        println!("     • Bias terms for input transformations");
        
        println!("  ➕ b_hh (Hidden biases): {}×{} = {} parameters", 
                 cell.b_hh.shape()[0], cell.b_hh.shape()[1],
                 cell.b_hh.shape()[0] * cell.b_hh.shape()[1]);
        println!("     • Bias terms for hidden transformations");
        
        println!("  📏 Hidden Size: {}", cell.hidden_size);
        println!();
    }
    
    println!("🧮 WHAT THESE PARAMETERS REPRESENT:");
    println!("  🚪 4 Gates per cell (each gets 1/4 of the weights):");
    println!("    • Input Gate (i): Decides what new information to store");
    println!("    • Forget Gate (f): Decides what to discard from memory");
    println!("    • Cell Gate (g): Creates candidate values to add");
    println!("    • Output Gate (o): Decides what parts of memory to output");
    println!();
}

fn inspect_file_contents() -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 ACTUAL FILE CONTENTS:");
    println!("========================");
    
    let file_size = fs::metadata("models/inspection_model.json")?.len();
    println!("  📊 File size: {} bytes", file_size);
    
    // Show a sample of the JSON structure
    let content = fs::read_to_string("models/inspection_model.json")?;
    let lines: Vec<&str> = content.lines().collect();
    
    println!("  📋 JSON Structure (first 30 lines):");
    for (i, line) in lines.iter().enumerate().take(30) {
        println!("    {}: {}", i + 1, line);
    }
    
    if lines.len() > 30 {
        println!("    ... ({} more lines)", lines.len() - 30);
    }
    println!();
    
    Ok(())
}

fn explain_llm_context() {
    println!("🤖 FOR LARGE LANGUAGE MODELS (LLMs):");
    println!("=====================================");
    println!("If you trained an LSTM-based LLM, the model would store:");
    println!();
    
    println!("📚 LEARNED LANGUAGE PATTERNS:");
    println!("  • Word relationships and dependencies");
    println!("  • Grammar and syntax patterns");
    println!("  • Semantic associations between concepts");
    println!("  • Long-term dependencies in text");
    println!();
    
    println!("🧠 HOW PATTERNS ARE ENCODED:");
    println!("  • w_ih weights: How input words influence gates");
    println!("    - Input gate: Which words trigger memory updates");
    println!("    - Forget gate: Which contexts cause forgetting");
    println!("    - Cell gate: What new information words contribute");
    println!("    - Output gate: What words trigger specific outputs");
    println!();
    
    println!("  • w_hh weights: How previous context influences current processing");
    println!("    - Encodes how past words affect current decisions");
    println!("    - Captures sequential dependencies and patterns");
    println!("    - Stores grammatical and syntactic relationships");
    println!();
    
    println!("  • Biases: Default tendencies and adjustments");
    println!("    - Language-specific biases (e.g., word order preferences)");
    println!("    - Default gate behaviors for the language");
    println!();
    
    println!("📖 EXAMPLE: Training on 'The cat sat on the mat':");
    println!("  • Weights learn that 'The' often precedes nouns");
    println!("  • 'cat' + 'sat' pattern gets encoded in w_hh");
    println!("  • 'on the' creates strong forward dependency");
    println!("  • Sentence boundaries trigger memory resets");
    println!();
    
    println!("💾 METADATA ALSO STORED:");
    println!("  • Training information (epochs, loss, date)");
    println!("  • Architecture details (vocabulary size, embedding dim)");
    println!("  • Model version and description");
    println!("  • Performance metrics");
    println!();
}

fn calculate_parameter_count(network: &LSTMNetwork) {
    println!("🔢 TOTAL PARAMETER COUNT:");
    println!("=========================");
    
    let mut total_params = 0;
    
    for (layer_idx, cell) in network.get_cells().iter().enumerate() {
        let w_ih_params = cell.w_ih.len();
        let w_hh_params = cell.w_hh.len();
        let b_ih_params = cell.b_ih.len();
        let b_hh_params = cell.b_hh.len();
        
        let layer_params = w_ih_params + w_hh_params + b_ih_params + b_hh_params;
        total_params += layer_params;
        
        println!("  Layer {}: {} parameters", layer_idx, layer_params);
    }
    
    println!("  🎯 Total: {} trainable parameters", total_params);
    println!("  💾 Memory: ~{:.1} KB (f64 precision)", total_params as f64 * 8.0 / 1024.0);
    println!();
    
    println!("📊 COMPARISON TO MODERN LLMs:");
    println!("  • GPT-3: ~175 billion parameters");
    println!("  • This LSTM: {} parameters", total_params);
    println!("  • Ratio: This model is {:.0}x smaller", 175_000_000_000.0 / total_params as f64);
    println!();
    
    println!("💡 SCALING FOR LLMs:");
    println!("  For a production LSTM LLM you might use:");
    println!("  • Input size: 512-1024 (embedding dimension)");
    println!("  • Hidden size: 1024-4096 (memory capacity)");
    println!("  • Layers: 6-12 (depth for complex patterns)");
    println!("  • Vocabulary: 50,000-100,000 tokens");
    
    // Calculate a realistic LLM size
    let llm_input = 512;
    let llm_hidden = 2048;
    let llm_layers = 8;
    
    let llm_params_per_layer = (llm_input * llm_hidden * 4) + (llm_hidden * llm_hidden * 4) + (llm_hidden * 4 * 2);
    let llm_total_params = llm_params_per_layer * llm_layers;
    
    println!("  Example LSTM LLM ({} layers, {} hidden): ~{:.1}M parameters", 
             llm_layers, llm_hidden, llm_total_params as f64 / 1_000_000.0);
} 