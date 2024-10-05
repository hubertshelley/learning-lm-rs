use crate::llm::module::Module;
use crate::llm::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Linear {
    tensor: Tensor,
}

impl Linear{
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> anyhow::Result<Tensor> {
        todo!()
    }
}