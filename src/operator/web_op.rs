use crate::cli::WebMode;
use crate::llm::model::Llama;
use anyhow::Result;
use tokenizers::Tokenizer;

pub(crate) fn operate(_mode: WebMode, llm: Llama, tokenizer: Tokenizer) -> Result<()> {
    Ok(())
}
