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
    // Unsigned integer using 32 bits.
    U32,
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
    pub fn default_data(&self, shape: &Vec<usize>) -> Vec<Data> {
        let length = shape.iter().product();
        match self {
            DType::BF16 => vec![bf16::default().into(); length],
            DType::F16 => vec![f16::default().into(); length],
            DType::F32 => vec![0.0f32.into(); length],
            DType::U32 => vec![0u32.into(); length],
        }
    }
    pub fn transfer_from_f32(&self, data: f32) -> Data {
        match self {
            DType::BF16 => bf16::from_f32(data).into(),
            DType::F16 => f16::from_f32(data).into(),
            DType::F32 => data.into(),
            DType::U32 => (data as u32).into(),
        }
    }
    pub fn transfer_from_usize(&self, data: usize) -> Data {
        match self {
            DType::BF16 => bf16::from_f32(data as f32).into(),
            DType::F16 => f16::from_f32(data as f32).into(),
            DType::F32 => (data as f32).into(),
            DType::U32 => (data as u32).into(),
        }
    }
}