use crate::cli::ChatMode;
use crate::llm::model::Llama;
use crate::llm::output;
use crate::types::completion::{ChatCompletionMessage, SystemMessage, UserMessage};
use anyhow::{anyhow, Result};
use std::io::stdin;
use std::sync::Arc;
use tera::{Context, Tera};
use tokenizers::Tokenizer;

pub(crate) fn operate(mode: ChatMode, llm: Llama, tokenizer: Tokenizer) -> Result<()> {
    let tera = Tera::new("templates/*")?;
    let mut messages: Vec<ChatCompletionMessage> = Vec::new();
    if let Some(system_prompt) = mode.system_prompt {
        messages.push(
            SystemMessage {
                content: system_prompt.clone(),
                ..Default::default()
            }
            .into(),
        );
    };
    let template = mode.template + ".jinja2";
    let mut input = String::new();
    eprint!("User: ");
    let llm = Arc::new(llm);
    if stdin().read_line(&mut input).is_ok() {
        messages.push(
            UserMessage {
                content: input.trim().to_string(),
                ..Default::default()
            }
            .into(),
        );
        let mut context = Context::new();
        context.insert("messages", &messages);
        context.insert("add_generation_prompt", &true);
        let input = tera.render(&template, &context)?;
        let binding = tokenizer
            .encode(input, true)
            .map_err(|e| anyhow!("Failed to encode prompt: {}", e))?;
        let input_ids = binding.get_ids();
        let output_ids = llm.generate(
            input_ids,
            mode.model_args.max_length as usize,
            mode.model_args.top_p,
            mode.model_args.top_k,
            mode.model_args.temperature,
        );
        let mut output = output::OutputGenerator::new(tokenizer.clone());
        eprint!("Assistant: ");
        if mode.model_args.stream {
            for token_id in output_ids {
                if let Some(token) = output.next_token(token_id) {
                    eprint!("{}", token);
                }
            }
        } else {
            eprint!(
                "{:?}",
                output.decode(output_ids.collect::<Vec<u32>>().as_slice())
            )
        }
        eprint!("\nUser: ");
    }
    while stdin().read_line(&mut input).is_ok() {
        let mut context = Context::new();
        let messages: Vec<ChatCompletionMessage> = vec![UserMessage {
            content: input.trim().to_string(),
            ..Default::default()
        }
        .into()];
        context.insert("messages", &messages);
        context.insert("add_generation_prompt", &true);
        let input = tera.render(&template, &context)?;
        let binding = tokenizer
            .encode(input, true)
            .map_err(|e| anyhow!("Failed to encode prompt: {}", e))?;
        let input_ids = binding.get_ids();
        let output_ids = llm.generate(
            input_ids,
            mode.model_args.max_length as usize,
            mode.model_args.top_p,
            mode.model_args.top_k,
            mode.model_args.temperature,
        );
        let mut output = output::OutputGenerator::new(tokenizer.clone());
        eprint!("Assistant: ");
        if mode.model_args.stream {
            for token_id in output_ids {
                if let Some(token) = output.next_token(token_id) {
                    eprint!("{}", token);
                }
            }
        } else {
            eprint!(
                "{:?}",
                output.decode(output_ids.collect::<Vec<u32>>().as_slice())
            )
        }
        eprint!("\nUser: ");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::types::completion::{ChatCompletionMessage, SystemMessage};
    use tera::{Context, Tera};

    #[test]
    fn test_jinja2_template() {
        let tera = Tera::new("templates/*")
            .map_err(|e| {
                eprintln!("Error parsing templates: {}", e);
                e
            })
            .unwrap();
        let mut messages: Vec<ChatCompletionMessage> = Vec::new();
        let system_prompt = "Hello, how can I help you?".to_string();
        messages.push(
            SystemMessage {
                content: system_prompt.clone(),
                ..Default::default()
            }
            .into(),
        );
        let mut context = Context::new();
        context.insert("messages", &messages);
        context.insert("add_generation_prompt", &true);
        let input = tera
            .render("chatml.jinja2", &context)
            .unwrap()
            .trim()
            .to_string();
        assert_eq!(
            input,
            "<|im_start|>system\nHello, how can I help you?<|im_end|>\n<|im_start|>assistant"
        );
    }
}
