use super::tensor::Tensor;
use crate::llm::dtype::DType;
pub struct KVCache {
    k_cache: Vec<Tensor>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl KVCache {
    pub fn new(
        n_layers: usize,
        max_seq_len: usize,
        dim: usize,
        init_len: usize,
        d_type: DType,
    ) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&[max_seq_len, dim], d_type))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&[max_seq_len, dim], d_type))
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor {
        self.k_cache[layer].slice(start * self.dim, &[self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor {
        self.v_cache[layer].slice(start * self.dim, &[self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }
}
