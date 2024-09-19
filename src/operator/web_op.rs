use crate::cli::WebMode;
use crate::llm::model::Llama;
use anyhow::Result;
use tokenizers::Tokenizer;

pub(crate) fn operate<T>(_mode: WebMode, llm: Llama<T>, tokenizer: Tokenizer) -> Result<()> {
    Ok(())
}
