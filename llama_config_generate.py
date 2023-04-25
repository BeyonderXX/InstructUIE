import json

config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
generation_config = {"_from_model_config": True, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "transformers_version": "4.27.0.dev0"}
tokenizer_config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}

with open("/mnt/data/user/zhou_weikang/model_cache/llama-7b-hf/config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False)

with open("/mnt/data/user/zhou_weikang/model_cache/llama-7b-hf/tokenizer_config.json", 'w', encoding='utf-8') as f:
    json.dump(tokenizer_config, f, ensure_ascii=False)

with open("/mnt/data/user/zhou_weikang/model_cache/llama-7b-hf/generation_config.json", 'w', encoding='utf-8') as f:
    json.dump(generation_config, f, ensure_ascii=False)