use crate::llm::linear::Linear;

#[derive(Debug, Clone)]
pub struct Mlp<T> {
    gate_proj: Linear<T>,
    up_proj: Linear<T>,
    down_proj: Linear<T>,
}