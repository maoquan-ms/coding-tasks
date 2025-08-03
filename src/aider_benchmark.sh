export OPENAI_API_BASE=http://localhost:33696/v1
export OPENAI_API_KEY=sk-no-key-required
# ./benchmark/benchmark.py Qwen --model openai/Qwen/Qwen2.5-Coder-0.5B-Instruct --languages python --edit-format whole --threads 10 --exercises-dir polyglot-benchmark

./benchmark/benchmark.py Qwen \
  --model openai/Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --languages python \
  --edit-format whole \
  --threads 8 \
  --exercises-dir polyglot-benchmark
#   --num-tests 10