use crate::llm::tensor::Tensor;
use anyhow::Result;

pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}