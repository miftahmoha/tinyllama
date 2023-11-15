import json

json_file_path = "./config/llama_config.json"

with open(json_file_path, "r") as json_file:
    DEFAULT_CONFIG = json.load(json_file)


model_config = DEFAULT_CONFIG["model"]

train_config = DEFAULT_CONFIG["train"]

lr_config = DEFAULT_CONFIG["lr_diagnosis"]

swiglu_config = DEFAULT_CONFIG["swiglu_diagnosis"]

gradient_config = DEFAULT_CONFIG["gradient_diagnosis"]

gdratio_config = DEFAULT_CONFIG["gdratio_diagnosis"]

gptune_config = DEFAULT_CONFIG["gptune"]
