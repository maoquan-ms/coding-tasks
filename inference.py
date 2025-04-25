
from vllm import LLM, SamplingParams
from human_eval.data import read_problems, write_jsonl


llm = LLM(model="./Qwen2.5-Coder-0.5B-Instruct", tensor_parallel_size=1)


sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512 )


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