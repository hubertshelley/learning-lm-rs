mod chat_op;
mod once_op;
mod server_op;
mod web_op;

use crate::cli::Mode;
use crate::llm::model::Llama;
use anyhow::Result;
use tokenizers::Tokenizer;

pub(crate) fn operate(llm: Llama<f32>, tokenizer: Tokenizer, mode: Mode) -> Result<()> {
    match mode {
        Mode::Once(mode) => {
            once_op::operate(mode, llm, tokenizer)?;
        }
        Mode::Chat(mode) => {
            chat_op::operate(mode, llm, tokenizer)?;
        }
        // Mode::Web(mode) => {
        //     web_op::operate(mode, llm, tokenizer)?;
        // }
        Mode::Server(mode) => {
            server_op::operate(mode, llm, tokenizer)?;
        }
    }
    Ok(())
}
