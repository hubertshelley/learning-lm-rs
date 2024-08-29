mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    // let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    //     let input = r#"<|im_start|>system
    // You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.<|im_end|>
    // <|im_start|>user
    // Hey! Got a question for you!<|im_end|>
    // <|im_start|>assistant
    // "#;
    let input = "<|start_story|>Once upon a time, ";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    eprint!("\n{}", input);
    let output_ids = llama.generate(input_ids, 256, 0.55, 35, 0.65);
    for token_id in output_ids {
        eprint!("{}", tokenizer.decode(&vec![token_id], true).unwrap());
    }
    // println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
