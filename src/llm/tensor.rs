use crate::llm::data::Data;
use crate::llm::dtype::DType;
use std::ops::Add;
use std::slice;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Tensor {
    data: Arc<Vec<Data>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
    pub d_type: DType,
}

impl Add<&Tensor> for Tensor {
    type Output = Self;
    fn add(self, other: &Tensor) -> Self {
        let left = self.data;
        let right = other.data.clone();
        let shape = self.shape;
        assert_eq!(shape, other.shape);
        assert_eq!(self.d_type, other.d_type);
        let length = self.length;
        Tensor {
            data: Arc::new(
                left.iter()
                    .cloned()
                    .zip(right.iter().cloned())
                    .map(|(x, y)| x + y)
                    .collect(),
            ),
            shape,
            offset: 0,
            length,
            d_type: self.d_type,
        }
    }
}

#[allow(dead_code)]
impl Tensor {
    pub fn default_data(&self) -> Data {
        self.data.first().unwrap().default_value()
    }
    pub fn new<T: Into<Data>>(data: Vec<T>, shape: &[usize]) -> Self {
        let data: Vec<Data> = data.into_iter().map(|x| x.into()).collect();
        let d_type = data.first().unwrap().d_type();
        let length = data.len();
        Tensor {
            data: Arc::new(data),
            shape: shape.to_owned(),
            offset: 0,
            length,
            d_type,
        }
    }

    pub fn data(&self) -> &[Data] {
        self.data.as_slice()
    }
    pub unsafe fn data_mut(&mut self) -> &mut [Data] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut Data;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn default(shape: &[usize], d_type: DType) -> Self {
        Self::new(d_type.default_data(shape), shape)
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

    pub fn slice(&self, start: usize, shape: &[usize]) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.to_owned(),
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

        a.iter().zip(b).all(|(x, y)| x == y)
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
