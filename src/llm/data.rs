use crate::llm::dtype::DType;
use half::{bf16, f16};
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Data {
    BF16(bf16),
    F16(f16),
    F32(f32),
    U32(u32),
}

impl Data {
    pub(crate) fn to_f32(self) -> f32 {
        match self {
            Data::BF16(v) => bf16::to_f32(v),
            Data::F16(v) => f16::to_f32(v),
            Data::F32(v) => v,
            Data::U32(v) => v as f32,
        }
    }

    pub(crate) fn total_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Data::BF16(l), Data::BF16(r)) => l.total_cmp(r),
            (Data::F16(l), Data::F16(r)) => l.total_cmp(r),
            (Data::F32(l), Data::F32(r)) => l.total_cmp(r),
            (Data::U32(l), Data::U32(r)) => (*l as f32).total_cmp(&(*r as f32)),
            _ => panic!("Cannot compare {:?} and {:?}", self, other),
        }
    }
}

impl Data {
    pub(crate) fn sin_cos(&self) -> (Data, Data) {
        match self {
            Data::BF16(v) => (
                Data::BF16(bf16::from_f32(v.to_f32().sin())),
                Data::BF16(bf16::from_f32(v.to_f32().cos())),
            ),
            Data::F16(v) => (
                Data::F16(f16::from_f32(v.to_f32().sin())),
                Data::F16(f16::from_f32(v.to_f32().cos())),
            ),
            Data::F32(v) => (Data::F32(v.sin()), Data::F32(v.cos())),
            Data::U32(v) => (
                Data::U32((*v as f32).sin() as u32),
                Data::U32((*v as f32).cos() as u32),
            ),
        }
    }
}

impl Add for Data {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l + r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l + r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l + r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l + r),
            _ => panic!("Cannot add {:?} and {:?}", self, rhs),
        }
    }
}

impl AddAssign for Data {
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self).add(rhs);
    }
}

impl Sub for Data {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l - r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l - r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l - r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l - r),
            _ => panic!("Cannot sub {:?} and {:?}", self, rhs),
        }
    }
}

impl Sub for &Data {
    type Output = <Data as Sub<Data>>::Output;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l - r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l - r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l - r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l - r),
            _ => panic!("Cannot sub {:?} and {:?}", self, rhs),
        }
    }
}

impl Mul for Data {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l * r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l * r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l * r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l * r),
            _ => panic!("Cannot mul {:?} and {:?}", self, rhs),
        }
    }
}

impl Mul for &Data {
    type Output = <Data as Mul<Data>>::Output;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l * r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l * r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l * r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l * r),
            _ => panic!("Cannot mul {:?} and {:?}", self, rhs),
        }
    }
}

impl Mul<f32> for Data {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            Data::BF16(v) => Data::BF16(bf16::from_f32(v.to_f32() * rhs)),
            Data::F16(v) => Data::F16(f16::from_f32(v.to_f32() * rhs)),
            Data::F32(v) => Data::F32(v * rhs),
            Data::U32(v) => Data::U32(v * rhs as u32),
        }
    }
}

impl Neg for Data {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Data::BF16(v) => Data::BF16(-v),
            Data::F16(v) => Data::F16(-v),
            Data::F32(v) => Data::F32(-v),
            Data::U32(v) => Data::U32(v),
        }
    }
}

impl Sum for Data {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Data::F32(0.0);
        for item in iter {
            if item.d_type() != sum.d_type() {
                sum = item.default_value();
            }
            sum += item;
        }
        sum
    }
}

impl Div<Data> for Data {
    type Output = Data;

    fn div(self, rhs: Data) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l / r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l / r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l / r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l / r),
            _ => panic!("Cannot div {:?} and {:?}", self, rhs),
        }
    }
}
impl Div<Data> for &Data {
    type Output = <Data as Div<Data>>::Output;

    fn div(self, rhs: Data) -> Self::Output {
        match (self, rhs) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l / r),
            (Data::F16(l), Data::F16(r)) => Data::F16(l / r),
            (Data::F32(l), Data::F32(r)) => Data::F32(l / r),
            (Data::U32(l), Data::U32(r)) => Data::U32(l / r),
            _ => panic!("Cannot div {:?} and {:?}", self, rhs),
        }
    }
}

impl From<u32> for Data {
    fn from(v: u32) -> Self {
        Data::U32(v)
    }
}

impl From<f32> for Data {
    fn from(v: f32) -> Self {
        Data::F32(v)
    }
}

impl From<f16> for Data {
    fn from(v: f16) -> Self {
        Data::F16(v)
    }
}

impl From<bf16> for Data {
    fn from(v: bf16) -> Self {
        Data::BF16(v)
    }
}

impl From<Data> for usize {
    fn from(v: Data) -> Self {
        match v {
            Data::BF16(v) => v.to_bits() as usize,
            Data::F16(v) => v.to_bits() as usize,
            Data::F32(v) => v as usize,
            Data::U32(v) => v as usize,
        }
    }
}

impl Data {
    pub fn d_type(&self) -> DType {
        match self {
            Data::BF16(_) => DType::BF16,
            Data::F16(_) => DType::F16,
            Data::F32(_) => DType::F32,
            Data::U32(_) => DType::U32,
        }
    }
    pub fn default_value(&self) -> Data {
        match self {
            Data::BF16(_) => Data::BF16(bf16::default()),
            Data::F16(_) => Data::F16(f16::default()),
            Data::F32(_) => Data::F32(0.0),
            Data::U32(_) => Data::U32(0),
        }
    }
    pub fn default_one(&self) -> Data {
        match self {
            Data::BF16(_) => Data::BF16(bf16::from_f32(1.0)),
            Data::F16(_) => Data::F16(f16::from_f32(1.0)),
            Data::F32(_) => Data::F32(1.0),
            Data::U32(_) => Data::U32(1),
        }
    }
    pub fn abs(&self) -> f32 {
        match self {
            Data::BF16(v) => v.to_f32().abs(),
            Data::F16(v) => v.to_f32().abs(),
            Data::F32(v) => v.abs(),
            Data::U32(v) => *v as f32,
        }
    }
    pub fn exp(&self) -> Data {
        match self {
            Data::BF16(v) => Data::BF16(bf16::from_f32(v.to_f32().exp())),
            Data::F16(v) => Data::F16(f16::from_f32(v.to_f32().exp())),
            Data::F32(v) => Data::F32(v.exp()),
            Data::U32(v) => Data::U32((*v as f32).exp() as u32),
        }
    }
    pub fn powi(&self, n: i32) -> Data {
        match self {
            Data::BF16(v) => Data::BF16(bf16::from_f32(v.to_f32().powi(n))),
            Data::F16(v) => Data::F16(f16::from_f32(v.to_f32().powi(n))),
            Data::F32(v) => Data::F32(v.powi(n)),
            Data::U32(v) => Data::U32(v.pow(n as u32)),
        }
    }
    pub(crate) fn powf(&self, n: f32) -> Data {
        match self {
            Data::BF16(v) => Data::BF16(bf16::from_f32(v.to_f32().powf(n))),
            Data::F16(v) => Data::F16(f16::from_f32(v.to_f32().powf(n))),
            Data::F32(v) => Data::F32(v.powf(n)),
            Data::U32(v) => Data::U32((*v as f32).powf(n) as u32),
        }
    }
    pub fn sqrt(&self) -> Data {
        match self {
            Data::BF16(v) => Data::BF16(bf16::from_f32(v.to_f32().sqrt())),
            Data::F16(v) => Data::F16(f16::from_f32(v.to_f32().sqrt())),
            Data::F32(v) => Data::F32(v.sqrt()),
            Data::U32(v) => Data::U32((*v as f32).sqrt() as u32),
        }
    }
    pub fn max(self, other: Self) -> Self {
        match (self, other) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l.max(r)),
            (Data::F16(l), Data::F16(r)) => Data::F16(l.max(r)),
            (Data::F32(l), Data::F32(r)) => Data::F32(l.max(r)),
            (Data::U32(l), Data::U32(r)) => Data::U32(l.max(r)),
            _ => panic!("Cannot max {:?} and {:?}", self, other),
        }
    }
    pub fn min(self, other: Self) -> Self {
        match (self, other) {
            (Data::BF16(l), Data::BF16(r)) => Data::BF16(l.min(r)),
            (Data::F16(l), Data::F16(r)) => Data::F16(l.min(r)),
            (Data::F32(l), Data::F32(r)) => Data::F32(l.min(r)),
            (Data::U32(l), Data::U32(r)) => Data::U32(l.min(r)),
            _ => panic!("Cannot min {:?} and {:?}", self, other),
        }
    }
}
