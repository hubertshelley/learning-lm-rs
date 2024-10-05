use crate::llm::module::Module;
use crate::llm::tensor::Tensor;

pub struct RmsNorm {
    tensor: Tensor,
}

impl RmsNorm {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> anyhow::Result<Tensor> {
        todo!()
    }
}