use crate::llm::config::LlamaConfigJson;
use crate::llm::linear::Linear;
use crate::llm::params::LLamaParams;

#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

impl CausalSelfAttention {
    pub fn new(
        params: LLamaParams<f32>,
        config: LlamaConfigJson, ) -> Self {
        Self {
            q_proj: params.wq.iter().cloned().map(|w| Linear::new(w)).collect(),
            k_proj: params.wk.iter().cloned().map(|w| Linear::new(w)).collect(),
            v_proj: params.wv.iter().cloned().map(|w| Linear::new(w)).collect(),
            o_proj: params.wo.iter().cloned().map(|w| Linear::new(w)).collect(),
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            max_position_embeddings: config.max_position_embeddings,
        }
    }
}