use super::tensor::Tensor;
use crate::llm::data::Data;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor, indices: &Tensor, table: &Tensor) {
    let length = indices.size();
    let table_shape = table.shape();
    assert_eq!(table_shape.len(), 2);
    let dim = table_shape[1];
    assert_eq!(y.size(), length * dim);
    for i in 0..length {
        let data: usize = indices.data()[i].into();
        let src = &table.data()[data * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor, start_pos: usize, theta: Data) {
    let shape = y.shape();
    assert_eq!(shape.len(), 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let d_data = y.d_type.transfer_from_usize(shape[2]);
    let d_type = y.d_type;
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = d_type.transfer_from_usize(start_pos + tok);
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let i_2 = d_type.transfer_from_usize(i * 2);
                let freq = pos / theta.powf((i_2 / d_data).to_f32());
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq = y.shape()[ndim - 1];
    let batch = y.size() / seq;
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let seq_data = data[b * seq..(b + 1) * seq].to_vec();
        let max = seq_data.iter().fold(seq_data[0], |a, b| a.max(*b));
        let sum = seq_data.iter().map(|&x| (x - max).exp()).sum::<Data>();
        for i in 0..seq {
            data[b * seq + i] = (seq_data[i] - max).exp() / sum;
        }
    }
}

pub fn rms_norm(y: &mut Tensor, x: &Tensor, w: &Tensor, epsilon: Data) {
    let len = y.size();
    let w_len = w.size();
    assert_eq!(y.size(), x.size());
    assert_eq!(len % w_len, 0);

    let w_len_data = x.d_type.transfer_from_usize(w_len);

    let y = unsafe { y.data_mut() };
    let x = x.data();
    let w = w.data();
    for i in 0..len / w_len {
        // 分母
        let denom = x[w_len * i..w_len * (i + 1)]
            .iter()
            .map(|&x| x.powi(2))
            .sum::<Data>()
            / w_len_data
            + epsilon;
        for j in 0..w_len {
            y[w_len * i + j] = w[j] * x[w_len * i + j] / denom.sqrt();
        }
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor, x: &Tensor) {
    let len = y.size();
    assert_eq!(len, x.size());

    let y = unsafe { y.data_mut() };
    let x = x.data();
    let one = x.first().unwrap().default_one();
    for (i, &item) in x.iter().enumerate() {
        y[i] = one / (one + (-item).exp()) * x[i] * y[i];
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor, beta: f32, a: &Tensor, b: &Tensor, alpha: f32) {
    let c_row = c.shape()[0];
    let c_col = c.shape()[1];
    assert_eq!(a.shape()[1], b.shape()[1]);
    assert_eq!(c_row, a.shape()[0]);
    assert_eq!(c_col, b.shape()[0]);
    let a_col = a.shape()[1];
    let b_col = b.shape()[1];
    let beta = c.d_type.transfer_from_f32(beta);
    let alpha = c.d_type.transfer_from_f32(alpha);
    let c = unsafe { c.data_mut() };
    let a = a.data();
    let b = b.data();
    for i in 0..c_row {
        for j in 0..c_col {
            let sum = a[i * a_col..i * a_col + a_col]
                .iter()
                .zip(b[j * b_col..j * b_col + b_col].iter())
                .map(|(&a, &b)| a * b)
                .sum::<Data>();
            c[i * c_col + j] = beta * c[i * c_col + j] + alpha * sum;
        }
    }
}
// C = beta * C + alpha * A @ B
// hint: You don't need to do an exp licit of B
pub fn matmul_b(c: &mut Tensor, beta: f32, a: &Tensor, b: &Tensor, alpha: f32) {
    let c_row = c.shape()[0];
    let c_col = c.shape()[1];
    assert_eq!(a.shape()[1], b.shape()[0]);
    assert_eq!(c_row, a.shape()[0]);
    assert_eq!(c_col, b.shape()[1]);
    let a_col = a.shape()[1];
    let b_col = b.shape()[1];
    let c = unsafe { c.data_mut() };
    let a = a.data();
    let b = b.data();
    for i in 0..c_row {
        for j in 0..c_col {
            let mut sum = c.first().unwrap().default_value();
            for x in a[i * a_col..i * a_col + a_col]
                .iter()
                .zip(b[j..].iter().step_by(b_col))
                .map(|(&a, &b)| a * b) {
                sum += x;
            }
            c[i * c_col + j] = c[i * c_col + j] * beta + sum * alpha;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor, y: &Tensor) -> Data {
    let len = x.size();
    assert_eq!(len, y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = x_.first().unwrap().default_value();
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert_eq!(x.shape()[x.shape().len() - 1], x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: Data,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &Data)> for Probability {
        #[inline]
        fn from((i, p): (usize, &Data)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, x.d_type.transfer_from_f32(1.));
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / x.d_type.transfer_from_f32(temperature)).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = x.d_type.transfer_from_f32(rand::random::<f32>()) * pk.min(pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::new(vec![2., 3., 4.].into(), &vec![1, 3]);
    let x = Tensor::new(vec![1., 2., 3.].into(), &vec![1, 3]);
    silu(&mut y, &x);
    println!("{:?}", y);
    assert!(y.close_to(
        &Tensor::new(vec![1.4621172, 5.2847824, 11.430889], &vec![1, 3]),
        1e-3,
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::new(vec![1., 2., 3., 4.].into(), &vec![2, 2]);
    let x = Tensor::new(vec![1., 2., 3., 4.].into(), &vec![2, 2]);
    let w = Tensor::new(vec![1., 2.].into(), &vec![2]);
    let epsilon = y.d_type.transfer_from_f32(1e-6);
    rms_norm(&mut y, &x, &w, epsilon);
    assert!(y.close_to(
        &Tensor::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416].into(),
            &vec![2, 2],
        ),
        1e-3,
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::new(vec![1., 2., 3., 4.].into(), &vec![2, 2]);
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.].into(), &vec![2, 3]);
    let b = Tensor::new(vec![1., 2., 3., 4., 5., 6.].into(), &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::new(vec![15., 34., 35., 81.].into(), &vec![2, 2]),
        1e-3,
    ));
}

#[test]
fn test_matmul_b() {
    let mut c = Tensor::new(vec![1., 2., 3., 4.].into(), &vec![2, 2]);
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.].into(), &vec![2, 3]);
    let b = Tensor::new(vec![1., 4., 2., 5., 3., 6.].into(), &vec![3, 2]);
    matmul_b(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::new(vec![15., 34., 35., 81.].into(), &vec![2, 2]),
        1e-3,
    ));
}
