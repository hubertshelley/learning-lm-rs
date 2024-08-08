use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();
            let size_in_bytes = tensor.dtype().size();
            let elem_count = data.len() / size_in_bytes;
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            let data: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count) };
            Tensor::new(data.to_vec(), &tensor.shape().to_vec())
        };
        let embedding_table = get_tensor("lm_head.weight");
        assert_eq!(
            embedding_table.shape(),
            &[config.vocab_size, config.hidden_size]
        );
        let rms_att_w = vec![
            get_tensor("model.layers.0.input_layernorm.weight"),
            get_tensor("model.layers.1.input_layernorm.weight"),
        ];
        assert_eq!(rms_att_w[0].shape(), &[config.hidden_size]);
        let wq = vec![
            get_tensor("model.layers.0.self_attn.q_proj.weight"),
            get_tensor("model.layers.1.self_attn.q_proj.weight"),
        ];
        // assert_eq!(
        //     wq[0].shape(),
        //     &[config.num_key_value_heads * config.num_attention_heads, config.hidden_size]
        // );
        let wk = vec![
            get_tensor("model.layers.0.self_attn.k_proj.weight"),
            get_tensor("model.layers.1.self_attn.k_proj.weight"),
        ];
        // assert_eq!(
        //     wk[0].shape(),
        //     &[
        //         config.num_key_value_heads * config.num_attention_heads,
        //         config.hidden_size
        //     ]
        // );
        let wv = vec![
            get_tensor("model.layers.0.self_attn.v_proj.weight"),
            get_tensor("model.layers.1.self_attn.v_proj.weight"),
        ];
        // assert_eq!(
        //     wv[0].shape(),
        //     &[
        //         config.num_key_value_heads * config.num_attention_heads,
        //         config.hidden_size
        //     ]
        // );
        let wo = vec![
            get_tensor("model.layers.0.self_attn.o_proj.weight"),
            get_tensor("model.layers.1.self_attn.o_proj.weight"),
        ];
        // assert_eq!(
        //     wv[0].shape(),
        //     &[
        //         config.hidden_size,
        //         config.num_key_value_heads * config.num_attention_heads,
        //     ]
        // );
        let rms_ffn_w = vec![
            get_tensor("model.layers.0.post_attention_layernorm.weight"),
            get_tensor("model.layers.1.post_attention_layernorm.weight"),
        ];
        assert_eq!(rms_ffn_w[0].shape(), &[config.hidden_size]);
        let w_up = vec![
            get_tensor("model.layers.0.mlp.up_proj.weight"),
            get_tensor("model.layers.1.mlp.up_proj.weight"),
        ];
        assert_eq!(
            w_up[0].shape(),
            &[config.intermediate_size, config.hidden_size]
        );
        let w_gate = vec![
            get_tensor("model.layers.0.mlp.gate_proj.weight"),
            get_tensor("model.layers.1.mlp.gate_proj.weight"),
        ];
        assert_eq!(
            w_gate[0].shape(),
            &[config.intermediate_size, config.hidden_size]
        );
        let w_down = vec![
            get_tensor("model.layers.0.mlp.down_proj.weight"),
            get_tensor("model.layers.1.mlp.down_proj.weight"),
        ];
        assert_eq!(
            w_down[0].shape(),
            &[config.hidden_size, config.intermediate_size]
        );
        let rms_out_w = get_tensor("model.norm.weight");
        assert_eq!(rms_out_w.shape(), &[config.hidden_size]);
        let lm_head = get_tensor("lm_head.weight");
        assert_eq!(lm_head.shape(), &[config.vocab_size, config.hidden_size]);

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
