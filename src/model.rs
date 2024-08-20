use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::operators::{masked_softmax, matmul_b, matmul_transb, rms_norm, silu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                self.n_q_h / self.n_kv_h,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            let shape = hidden_states.shape().clone();
            let x = hidden_states.slice(0, &shape);
            OP::matmul_transb(&mut hidden_states, 0., &x, &self.params.wo[layer], 1.0);
            // todo!("down_proj matmul and add residual");

            // todo!("mlp(...)");
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_att_w[layer],
                self.eps,
            );
            // break;
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();

        // todo!("实现文本生成");
        let mut cache = self.new_cache();
        let mut logits = Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]);
        for index in 0..self.max_seq_len {
            let logits_f32 = self.forward(&logits, &mut cache);
            let sample = OP::random_sample(&logits_f32, top_p, top_k, temperature);
            // break;
            if sample == self.eos_token_id {
                break;
            }

            result.push(sample);

            if index == max_len {
                break;
            }

            let mut data = logits.data().to_vec();
            data.push(sample);
            logits = Tensor::<u32>::new(vec![sample], &vec![1]);
        }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // 数据检查
    assert_eq!(
        hidden_states.shape(),
        &vec![seq_len, n_kv_h * n_groups * dqkv]
    );
    assert_eq!(
        att_scores.shape(),
        &vec![n_kv_h, n_groups, seq_len, total_seq_len]
    );
    assert_eq!(q.shape(), &vec![seq_len, n_kv_h * n_groups, dqkv]);
    assert_eq!(k.shape(), &vec![total_seq_len, n_kv_h * dqkv]);
    assert_eq!(v.shape(), &vec![total_seq_len, n_kv_h * dqkv]);

    let mut att_scores_list = {
        let mut items = Vec::new();
        for i in 0..n_kv_h {
            items.push(att_scores.slice(
                i * n_groups * seq_len * total_seq_len,
                &vec![n_groups * seq_len, total_seq_len],
            ));
        }
        items
    };

    let q_list = {
        let mut items = Vec::new();
        for i in 0..n_kv_h {
            items.push(q.slice(
                i * n_groups * seq_len * dqkv,
                &vec![n_groups * seq_len, dqkv],
            ));
        }
        items
    };

    let k_list = {
        let mut items = Vec::new();
        for i in 0..n_kv_h {
            items.push(k.slice(i * total_seq_len * dqkv, &vec![total_seq_len, dqkv]));
        }
        items
    };

    let v_list = {
        let mut items = Vec::new();
        for i in 0..n_kv_h {
            items.push(v.slice(i * total_seq_len * dqkv, &vec![total_seq_len, dqkv]));
        }
        items
    };

    for i in 0..n_kv_h {
        matmul_transb(
            &mut att_scores_list[i],
            0.,
            &q_list[i],
            &k_list[i],
            1.0 / (dqkv as f32).sqrt(),
        )
    }

    let att_scores_data = unsafe { att_scores.data_mut() };
    for (i, &f) in att_scores_list.iter().flat_map(|x| x.data()).enumerate() {
        att_scores_data[i] = f;
    }
    masked_softmax(att_scores);

    let att_scores_list = {
        let mut items = Vec::new();
        for i in 0..n_kv_h {
            items.push(att_scores.slice(
                i * n_groups * seq_len * total_seq_len,
                &vec![n_groups * seq_len, total_seq_len],
            ));
        }
        items
    };
    let mut hidden_states_list = {
        let mut items = Vec::new();
        for _ in 0..n_kv_h {
            items.push(Tensor::<f32>::default(&vec![n_groups * seq_len, dqkv]));
        }
        items
    };

    for i in 0..n_kv_h {
        matmul_b(
            &mut hidden_states_list[i],
            0.,
            &att_scores_list[i],
            &v_list[i],
            1.0,
        )
    }
    let hidden_states_data = unsafe { hidden_states.data_mut() };
    for (i, &f) in hidden_states_list.iter().flat_map(|x| x.data()).enumerate() {
        hidden_states_data[i] = f;
    }

    // matmul_transb(
    //     hidden_states,
    //     0.,
    //     att_scores.reshape(&vec![att_scores.size() / (n_kv_h * dqkv), n_kv_h * dqkv]),
    //     v,
    //     1.0,
    // )
    // vec_multi_wight(hidden_states, att_scores, v);
    // let mut hidden_states_data = unsafe { hidden_states.data_mut() };
    // hidden_states_data
    //     .iter_mut()
    //     .enumerate()
    //     .for_each(|(i, &mut mut x)| x += att_scores.data()[i]);
}
pub fn vec_multi(c: &mut Tensor<f32>, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32, t: bool) {
    // 判断c，长度是否大于二
    assert!(
        c.shape().len() > 2,
        "vec_multi of dimensions must be at least 2"
    );
    // a 重要，用于切分数据
    assert!(a.shape().len() == 2, "vec_multi of dimensions must be 2");
    assert!(b.shape().len() == 2, "vec_multi of dimensions must be 2");
    let shape = c.shape();
    // 获取矩阵的行列数
    let (row, column) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    // 获取n_q_h，用于分组
    let q_head_len = shape[..shape.len() - 2].iter().product::<usize>();
    // 确定qk的倍数对应关系
    let q_k_reflect = a.shape()[1] / b.shape()[1];
    let vec_len = a.shape()[1] / q_head_len;
    let a_data = a.data();
    // 用于获取q_head需要进行跳过的数值
    let a_skip = a.shape()[1];
    let b_data = b.data();
    // 用于获取k_head需要进行跳过的数值
    let b_skip = b.shape()[1];
    let data = unsafe { c.data_mut() };
    // 清理脏数据
    data.fill(0.);
    let mut c_data_offset = 0;
    if t {
        // 用于分组计算，每个输入，在每个请求头下的vjiv
        for i in 0..q_head_len {
            // 计算一个输入值，在一个请求头下的total中的所有v
            for j in 0..row {
                // 临时q_head 值,j*a_skip用于跳过多头i*16用于跳过单头
                let a_tmp =
                    &a_data[(i * vec_len + j * a_skip)..(i * vec_len + j * a_skip) + vec_len];
                // 计算单一v
                for k in 0..column {
                    let b_tmp = &b_data[(k * b_skip + (i / q_k_reflect) * vec_len)
                        ..(k * b_skip + (i / q_k_reflect) * vec_len) + vec_len];
                    data[c_data_offset] = a_tmp
                        .iter()
                        .zip(b_tmp.iter())
                        .fold(0., |tmp, (a_val, b_val)| tmp + a_val * b_val)
                        * alpha;
                    c_data_offset += 1;
                }
            }
        }
    }
}
// 只用于得分计算
// a代表所处理的权重,b代表所要乘的向量
pub fn vec_multi_wight(c: &mut Tensor<f32>, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert!(
        b.shape().len() == 2,
        "matmul_transb of dimensions must be at least 2"
    );
    assert!(
        a.shape().len() == 4,
        "matmul_transb of dimensions must be  4 是att_scores)"
    );
    let q_header_len = a.shape()[..a.shape().len() - 2].iter().product::<usize>();
    let shape = a.shape();
    // 获取矩阵的行列数
    let (row, column) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    // 获取计算向量的长度
    let vec_len = b.shape()[1] / a.shape()[0];
    // 确认a，b需要的对应关系,默认a的长度大于b的长度
    let n_groups = a.shape()[1];
    let b_column = b.shape()[1];
    let mut data = unsafe { c.data_mut() };
    // 清理脏数据
    data.fill(0.);
    for i in 0..q_header_len {
        // 获取当前q下的的全部注意力
        let a_data = &a.data()[i * row * column..(i + 1) * row * column];
        // 循环计算每个当前q下，每个输入的v权重
        for c_i in 0..row {
            // 用于标记当前计算到那一列
            let mut b_data_row_offset = 0;
            let tmp_c_offset = n_groups * b_column * c_i + i * vec_len;
            // 获取c存储当先向量的位置，
            let tmp_c = &mut data[tmp_c_offset..tmp_c_offset + vec_len];
            // 获取一个输入的全部注意力
            a_data[c_i * column..(c_i + 1) * column]
                .iter()
                .for_each(|tmp| {
                    // 获取q，对应的v b_data_row_offset*b_column表示要跳过的input
                    // (q_header_len/n_groups)*vec_len 表示q对应的v
                    let tmp_offset = b_data_row_offset * b_column + (i / n_groups) * vec_len;
                    let b_data = &b.data()[tmp_offset..tmp_offset + vec_len];
                    b_data.iter().zip(tmp_c.iter_mut()).for_each(|(t_b, t_c)| {
                        *t_c += t_b * tmp;
                    });
                    // 进行偏移
                    b_data_row_offset += 1;
                });
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0., hidden_states, w_gate, 1.0);
    matmul_transb(up, 0., hidden_states, w_up, 1.0);
    silu(up, gate);
    matmul_transb(hidden_states, 0., up, w_down, 1.0);
    let residual = unsafe { residual.data_mut() };
    for i in 0..residual.len() {
        residual[i] += hidden_states.data()[i];
    }
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
