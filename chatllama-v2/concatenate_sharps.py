############################################
## combine multiple weight files into one ##
############################################

import os
import sys
import torch
import json


input_base_path = "/root/MODELS/llama_13B"
with open(os.path.join(input_base_path, "13B/params.json"), "r") as file:   
    params = json.load(file)
num_shards = 2 # 13B
n_layers = params["n_layers"]

loaded = [
    torch.load(os.path.join(input_base_path, f"13B/consolidated.{i:02d}.pth"), map_location="cpu")
    for i in range(num_shards)
]

state_dict = {
    "tok_embeddings.weight": torch.cat(
        [
            loaded[i]["tok_embeddings.weight"] 
            for i in range(num_shards)
        ],
        dim=1
    ),
    "norm.weight": loaded[0]["norm.weight"],
    "output.weight": torch.cat(
        [
            loaded[i]["output.weight"]
            for i in range(num_shards)
        ],
        dim=0
    ),
    "rope.freqs": loaded[0]["rope.freqs"]
}

for layer_i in range(n_layers):
    state_dict[f"layers.{layer_i}.attention_norm.weight"] = loaded[0][f"layers.{layer_i}.attention_norm.weight"]
    state_dict[f"layers.{layer_i}.ffn_norm.weight"] = loaded[0][f"layers.{layer_i}.ffn_norm.weight"]
    state_dict[f"layers.{layer_i}.attention.wq.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.attention.wq.weight"]
            for i in range(num_shards)
        ],
        dim=0
    )
    state_dict[f"layers.{layer_i}.attention.wk.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.attention.wk.weight"]
            for i in range(num_shards)
        ],
        dim=0
    )
    state_dict[f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.attention.wv.weight"]
            for i in range(num_shards)
        ],
        dim=0
    )
    state_dict[f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.attention.wo.weight"]
            for i in range(num_shards)
        ],
        dim=1
    )
    state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] 
            for i in range(num_shards)
        ], 
        dim=0
    )
    state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] 
            for i in range(num_shards)
        ], 
        dim=1
    )
    state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
        [
            loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] 
            for i in range(num_shards)
        ], 
        dim=0
    )

torch.save(state_dict, os.path.join(input_base_path, "consolidated.00.pth"))
    