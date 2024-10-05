use std::ops::Add;
use half::{bf16, f16};
use crate::llm::dtype::DType;

#[derive(Debug, Clone)]
pub enum Data {
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
}

impl Add for Data {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => {
                Data::BF16(l.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            }
            (Data::F16(l), Data::F16(r)) => {
                Data::F16(l.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            }
            (Data::F32(l), Data::F32(r)) => {
                Data::F32(l.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            }
            _ => panic!("Cannot add {:?} and {:?}", self, rhs),
        }
    }
}

impl From<Vec<f32>> for Data {
    fn from(v: Vec<f32>) -> Self {
        Data::F32(v)
    }
}

impl From<Vec<f16>> for Data {
    fn from(v: Vec<f16>) -> Self {
        Data::F16(v)
    }
}

impl From<Vec<bf16>> for Data {
    fn from(v: Vec<bf16>) -> Self {
        Data::BF16(v)
    }
}

impl Data {
    pub fn d_type(&self) -> DType {
        match self {
            Data::BF16(_) => DType::BF16,
            Data::F16(_) => DType::F16,
            Data::F32(_) => DType::F32,
        }
    }
}