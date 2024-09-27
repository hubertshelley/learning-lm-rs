use crate::cli::OnceMode;
use crate::llm::{model::Llama, output};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub(crate) fn operate(mode: OnceMode, llm: Llama<f32>, tokenizer: Tokenizer) -> Result<()> {
    let binding = tokenizer
        .encode(mode.prompt.clone(), true)
        .map_err(|e| anyhow!("Failed to encode prompt: {}", e))?;
    let input_ids = binding.get_ids();
    let output_ids = Arc::new(llm).generate(
        input_ids,
        mode.model_args.max_length as usize,
        mode.model_args.top_p,
        mode.model_args.top_k,
        mode.model_args.temperature,
    );
    let mut output = output::OutputGenerator::new(tokenizer.clone());
    eprint!("{}", mode.prompt);
    if mode.model_args.stream {
        for token_id in output_ids {
            if let Some(token) = output.next_token(token_id) {
                eprint!("{}", token);
            }
        }
    } else {
        println!(
            "{:?}",
            output.decode(output_ids.collect::<Vec<u32>>().as_slice())
        )
    }
    Ok(())
}
