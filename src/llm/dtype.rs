use anyhow::anyhow;
use half::{bf16, f16};
use crate::llm::data::Data;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    // Brain floating-point using half precision (16 bits).
    BF16,
    // Floating-point using half precision (16 bits).
    F16,
    // Floating-point using single precision (32 bits).
    F32,
}

impl std::str::FromStr for DType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "float32" => Ok(Self::F32),
            "bfloat16" => Ok(Self::BF16),
            "float16" => Ok(Self::F16),
            _ => Err(anyhow!("Invalid data type: {}", s)),
        }
    }
}

impl DType {
    pub fn default_data(&self, shape: &Vec<usize>) -> Data {
        let length = shape.iter().product();
        match self {
            DType::BF16 => Data::BF16(vec![bf16::default(); length]),
            DType::F16 => Data::F16(vec![f16::default(); length]),
            DType::F32 => Data::F32(vec![0.0; length]),
        }
    }
}