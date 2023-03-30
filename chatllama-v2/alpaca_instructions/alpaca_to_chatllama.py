import json

with open("alpaca_instructions/alpaca_data.json", "r") as file:
    dataset = json.load(file)

newset = []
for item in dataset:
    user_input = item["instruction"]
    if item["input"] != "":
        user_input += ("\n" + item["input"])
    user_input += "\nAnswer: "
    if user_input.find("\t") >= 0:
        continue # Not support \t in instructions
    completion = item["output"]
    newset.append({"user_input": user_input, "completion": completion})
print(len(newset))

with open("alpaca_instructions/alpaca_instructions.json", "w") as file:
    json.dump(newset, file)
    