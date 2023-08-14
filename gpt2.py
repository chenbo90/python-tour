from transformers import GPT2Config, GPT2Model

# Initializing a GPT2 configuration
configuration = GPT2Config()
print(configuration)

'''
GPT2Config {
  "activation_function": "gelu_new",
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "transformers_version": "4.24.0",
  "use_cache": true,
  "vocab_size": 50257
}
'''

# Initializing a model (with random weights) from the configuration
model = GPT2Model(configuration)
print(model)

# Accessing the model configuration
configuration = model.config
print(configuration)