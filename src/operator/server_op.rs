use crate::cli::ServerMode;
use crate::llm::model::Llama;
use std::net::SocketAddr;
use std::sync::Arc;
use anyhow::anyhow;
use async_trait::async_trait;
use tokenizers::Tokenizer;
use silent::prelude::*;
use tera::{Context, Tera};
use crate::llm::output;
use crate::types::completion::{AssistantMessage, ChatCompletionChoice, ChatResponseFormat, UserMessage};
use crate::types::request::ChatCompletionRequest;
use crate::types::response::ChatCompletionResponse;

pub(crate) fn operate(mode: ServerMode, llm: Llama<f32>, tokenizer: Tokenizer) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{}:{}", mode.host, mode.port).parse()?;
    let llm = Arc::new(llm);
    let server = Server::new().bind(addr);
    let middleware = LlmMiddleware { llm, tokenizer, template: mode.template + ".jinja2", tera: Tera::new("templates/*")? };
    let route = Route::new("/v1/chat/completions").hook(middleware).post(chat_completions);
    server.run(route);
    Ok(())
}

struct LlmMiddleware {
    llm: Arc<Llama<f32>>,
    tokenizer: Tokenizer,
    template: String,
    tera: Tera,
}

#[async_trait]
impl MiddleWareHandler for LlmMiddleware {
    async fn pre_request(&self, req: &mut Request, _res: &mut Response) -> Result<MiddlewareResult> {
        req.configs_mut().insert(self.llm.clone());
        req.configs_mut().insert(self.tokenizer.clone());
        req.configs_mut().insert(self.template.clone());
        req.configs_mut().insert(self.tera.clone());
        Ok(MiddlewareResult::Continue)
    }
}

pub(crate) async fn chat_completions(mut req: Request) -> Result<Response> {
    let chat_completion_req: ChatCompletionRequest = req.json_parse().await?;
    let llm = req.get_config::<Arc<Llama<f32>>>()?;
    let tokenizer = req.get_config::<Tokenizer>()?;
    let template = req.get_config::<String>()?;
    let tera = req.get_config::<Tera>()?;

    let ChatCompletionRequest {
        messages,
        temperature,
        top_p,
        max_tokens,
        ..
    } = chat_completion_req;

    let mut context = Context::new();
    context.insert("messages", &messages);
    context.insert("add_generation_prompt", &true);
    let input = tera.render(&template, &context).map_err(|e| anyhow!("Failed to render jinja2 template: {}", e))?;
    let binding = tokenizer.encode(input, true).map_err(|e| anyhow!("Failed to encode prompt: {}", e))?;
    let input_ids = binding.get_ids();
    let prompt_tokens = input_ids.len();
    let output_ids = llm.generate(input_ids, max_tokens.unwrap_or(llm.max_seq_len), top_p.unwrap_or(1.0), 40, temperature.unwrap_or(0.9));
    let mut output = output::OutputGenerator::new(tokenizer.clone());

    if chat_completion_req.stream.clone().unwrap_or(false) {
        // let stream = chat_model.stream_handle(chat_completion_req).map_err(|e| {
        //     SilentError::business_error(
        //         StatusCode::BAD_REQUEST,
        //         format!("failed to handle chat model: {}", e),
        //     )
        // })?;
        // let result = sse_reply(stream);
        Ok(Response::empty())
    } else {
        let mut response = ChatCompletionResponse::new("".to_string());
        let mut choice = ChatCompletionChoice {
            finish_reason: Default::default(),
            index: 0,
            message: AssistantMessage {
                content: Some("".to_string()),
                name: None,
                tool_calls: vec![],
            },
        };
        let mut result = String::new();
        let mut sampled = 0;
        for token_id in output_ids {
            if let Some(token) = output.next_token(token_id) {
                result += &token;
            }
            sampled += 1;
        }
        choice.message.content = Some(result);
        response.choices.push(choice);
        response.usage.prompt_tokens = prompt_tokens;
        response.usage.completion_tokens = sampled;
        response.usage.total_tokens = prompt_tokens + sampled;
        match chat_completion_req.response_format {
            None => Ok(response.into()),
            Some(format) => {
                if format.r#type == ChatResponseFormat::Json {
                    Ok(response.into())
                } else {
                    let result = match response.choices.first() {
                        None => "".to_string(),
                        Some(choice) => choice.message.content.clone().unwrap_or("".to_string()),
                    };
                    Ok(result.into())
                }
            }
        }
    }
}
