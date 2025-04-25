# from datasets import load_dataset
# from vllm import LLM, SamplingParams
# import jsonlines

# dataset = load_dataset("./openai_humaneval")
# test_dataset = dataset["test"]

# model = LLM(
#     model="./Qwen2.5-Coder-0.5B-Instruct",
#     tensor_parallel_size=1
# )

# sampling_params = SamplingParams(
#     temperature=0.6, 
#     top_p=0.95, 
#     max_tokens=1024  
# )

# # with jsonlines.open("humaneval_predictions.jsonl", "w") as writer:
# #     for example in test_dataset:
# #         prompt = example["prompt"]
# #         outputs = model.generate([prompt], sampling_params)
# #         generated_text = outputs[0].outputs[0].text
        
     
# #         if "```python" in generated_text:
# #             generated_text = generated_text.split("```python", 1)[1].strip()
# #         generated_text = generated_text.split("\n\n#", 1)[0].strip()  # 移除测试用例
        
# #         writer.write({
# #             "task_id": example["task_id"],
# #             "completion": generated_text,
# #             "prompt": example["prompt"]
# #         })


# from human_eval.data import write_jsonl, read_problems

# problems = read_problems()

# num_samples_per_task = 200
# samples = [
#     dict(task_id=task_id, completion=model.generate(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("samples.jsonl", samples)


# inference_humaneval.py
from vllm import LLM, SamplingParams
from human_eval.data import read_problems, write_jsonl

# 初始化LLM，使用GPU
llm = LLM(model="./Qwen2.5-Coder-0.5B-Instruct", tensor_parallel_size=1)

# 定义采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512 )

# 读取HumanEval问题
problems = read_problems()

num_samples_per_task = 1
samples = []
for task_id in problems:
    prompt = problems[task_id]["prompt"]
    outputs = llm.generate(prompt, sampling_params)
    for _ in range(num_samples_per_task):
        completion = outputs[0].outputs[0].text
        sample = dict(task_id=task_id, completion=completion)
        samples.append(sample)

# 将生成的样本保存到JSON Lines文件
write_jsonl("samples.jsonl", samples)