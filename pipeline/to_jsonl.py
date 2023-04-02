import os
import json
root = '/mnt/bd/zenithbloomvol/data/IE_data_v3'
train_fpath = os.path.join(root, 'UIE_train.json')
train_out_fpath = os.path.join(root, 'UIE_train.jsonl')
dev_fpath = os.path.join(root, 'UIE_dev.json')
dev_out_fpath = os.path.join(root, 'UIE_dev.jsonl')



with open(train_fpath, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(' * Writing into', train_out_fpath)
with open(train_out_fpath, 'w', encoding='utf-8') as f:
    for item in train_data:
        line = json.dumps(
            {
                'inputs': item['user_input'],
                'targets': item['completion']
            }
        )
        line += '\n'
        f.write(line)

with open(dev_fpath, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

print(' * Writing into', dev_out_fpath)
with open(dev_out_fpath, 'w', encoding='utf-8') as f:
    for item in dev_data:
        line = json.dumps(
            {
                'inputs': item['user_input'],
                'targets': item['completion']
            }
        )
        line += '\n'
        f.write(line)