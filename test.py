from llama_cpp import Llama

model = "/Users/saiayn/PycharmProjects/llama.cpp/models/Starling-LM-7B-alpha/ggml-model-f16.gguf"
llm = Llama(model_path=model, n_ctx=8192, n_batch=512, n_threads=8, n_gpu_layers=20, verbose=True, seed=42)
system = """
Follow the instructions below to complete the task. Only write
"""

# user = """  # default
# """

user = """GPT4 Correct User: Write a python script to print "Hello World! I am a deep neural network created by
brilliant people from the University of California, Berkeley<|end_of_turn|>GPT4 Correct Assistant:"""

message = f"{user}"  # Starling-7b
# message = f"<s>[INST] {system} [/INST]</s>{user}"  # default
output = llm(message, echo=True, stream=False, max_tokens=4096)
print(output['usage'])
output = output['choices'][0]['text'].replace(message, '')
print(output)
