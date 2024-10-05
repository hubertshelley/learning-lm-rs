use crate::llm::data::Data;
use crate::llm::dtype::DType;
use std::ops::Add;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Tensor {
    data: Arc<Box<Data>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
    d_type: DType,
}

impl Add<&Tensor> for Tensor
{
    type Output = Self;
    fn add(self, other: &Tensor) -> Self {
        let left = self.data;
        let right = other.data.clone();
        let shape = self.shape;
        assert_eq!(shape, other.shape);
        assert_eq!(self.d_type, other.d_type);
        let length = self.length;
        let mut data = Vec::new();
        for i in 0..length {
            data.push(left[i] + right[i]);
        }
        Tensor {
            data: Arc::new(Box::new(left.add(***&right))),
            shape,
            offset: 0,
            length,
            d_type: self.d_type,
        }
    }
}

impl Tensor {
    pub fn new(data: Data, shape: &Vec<usize>) -> Self {
        let d_type = data.d_type();
        let length = data.len();
        Tensor {
            data: Arc::new(Box::new(data)),
            shape: shape.clone(),
            offset: 0,
            length,
            d_type,
        }
    }

    pub fn default(shape: &Vec<usize>, d_type: DType) -> Self {
        Self::new(d_type.default_data(&shape), shape)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!(
                "New shape {new_shape:?} does not match tensor of {:?}",
                old_shape
            );
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
            d_type: self.d_type,
        }
    }
}

// Some helper functions for testing and debugging
impl Tensor {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel))
    }
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
