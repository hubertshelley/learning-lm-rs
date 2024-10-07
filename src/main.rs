pub(crate) mod cli;
pub(crate) mod llm;
mod operator;
#[allow(dead_code)]
mod types;

use crate::cli::Cli;
use clap::Parser;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let cli = Cli::parse();
    let model_dir = PathBuf::from(cli.model_path);
    let llama = llm::model::Llama::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    operator::operate(llama, tokenizer, cli.mode).expect("Error running operator");
}
// fn main() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("chat");
//     // let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
//     let input = r#"<|im_start|>system
// You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.<|im_end|>
// <|im_start|>user
// Hey! Got a question for you!<|im_end|>
// <|im_start|>assistant
// "#;
//     // let input = "<|start_story|>Once upon a time, ";
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();
//     eprint!("\n{}", input);
//     let output_ids = llama.generate(input_ids, 256, 0.55, 35, 0.65);
//     let mut output = output::OutputGenerator::new(tokenizer.clone());
//
//     // let output_ids = output_ids.into_iter().collect::<Vec<_>>();
//     // // eprint!("{}", tokenizer.decode(&output_ids, true).unwrap());
//     for token_id in output_ids {
//         if let Some(token) = output.next_token(token_id) {
//             eprint!("{}", token);
//         }
//     }
//     // println!("{}", tokenizer.decode(&output_ids, true).unwrap());
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    fn test_once_forward() {
        let mode = cli::OnceMode {
            prompt: "Once upon a time, ".to_string(),
            model_args: cli::ModelArgs {
                max_length: 512,
                top_k: 1,
                temperature: 1.0,
                top_p: 1.0,
                stream: true,
            },
        };
        let model_dir = PathBuf::from("./models/story");
        let llama = llm::model::Llama::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let binding = tokenizer.encode(mode.prompt.clone(), true).unwrap();
        let input_ids = binding.get_ids();
        let output_ids = Arc::new(llama).generate(
            input_ids,
            mode.model_args.max_length as usize,
            mode.model_args.top_p,
            mode.model_args.top_k,
            mode.model_args.temperature,
        );
        for _token_id in output_ids {
            return;
        }
    }
}
