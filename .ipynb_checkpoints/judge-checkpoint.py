from human_eval.evaluation import evaluate_functional_correctness

results = evaluate_functional_correctness(
    sample_file="./samples.jsonl",
    problem_file="./data/HumanEval.jsonl", 
    k=[1],         
    n_workers=4,    
    
)

print(f"Pass@1: {results['pass@1'] * 100:.1f}%")
