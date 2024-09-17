use crate::cli::WebMode;
use anyhow::Result;
use tokenizers::Tokenizer;
use crate::llm::model::Llama;

pub(crate) fn operate<T>(_mode: WebMode, llm: Llama<T>, tokenizer: Tokenizer) -> Result<()> {
    Ok(())
}