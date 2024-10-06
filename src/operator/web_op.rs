use crate::cli::WebMode;
use crate::llm::model::Llama;
use anyhow::Result;
use tokenizers::Tokenizer;

#[allow(dead_code)]
pub(crate) fn operate(_mode: WebMode, _llm: Llama, _tokenizer: Tokenizer) -> Result<()> {
    Ok(())
}
