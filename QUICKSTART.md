# LLMRouter Quick Start

This is the shortest path to verify the project works from a fresh checkout.

## 1. Create a Python environment

```bash
cd /config/workspace/LLMRouter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Notes:
- The project requires Python 3.10+.
- `pip install -e .` is required before running the CLI or tests from a clean clone.

## 2. Verify the CLI loads

```bash
python -m llmrouter.cli.router_main --help
```

You should see the available subcommands:
- `train`
- `infer`
- `chat`
- `list-routers`
- `version`
- `serve`

## 3. Run the simplest smoke test

Use the built-in example config and `--route-only` so no LLM API call is made.

```bash
python -m llmrouter.cli.router_main infer \
  --router smallest_llm \
  --config configs/model_config_test/smallest_llm.yaml \
  --query "What is AI?" \
  --route-only
```

Expected result:
- JSON output on stdout
- `success: true`
- a selected `model_name`

This is the fastest end-to-end check because it uses bundled example data and does not require training.

## 4. Optional: test another zero-training router

```bash
python -m llmrouter.cli.router_main infer \
  --router largest_llm \
  --config configs/model_config_test/largest_llm.yaml \
  --query "Explain transformers" \
  --route-only
```

## 5. If you want an actual model response

Remove `--route-only`, but first set `API_KEYS`.

Example:

```bash
export API_KEYS='{"NVIDIA":"your-nvidia-api-key"}'
python -m llmrouter.cli.router_main infer \
  --router smallest_llm \
  --config configs/model_config_test/smallest_llm.yaml \
  --query "What is AI?"
```

## 6. Run LLMRouter as an OpenAI-compatible router for OpenClaw

Use the separate budget-aware serve config at [openclaw_router/config-budget-aware.yaml](/config/workspace/LLMRouter/openclaw_router/config-budget-aware.yaml).

Set your NVIDIA key in the same shell where you will start the router:

```bash
export NVIDIA_API_KEY='your-nvidia-api-key'
```

Start the router server from the repo root:

```bash
cd /config/workspace/LLMRouter
llmrouter serve --config openclaw_router/config-budget-aware.yaml
```

If `llmrouter` is not on `PATH`, use the module form:

```bash
cd /config/workspace/LLMRouter
python3 -m llmrouter.cli.router_main serve --config openclaw_router/config-budget-aware.yaml
```

The router will listen at:

```text
http://127.0.0.1:8000/v1
```

Configure OpenClaw by merging this provider into `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "openclaw": {
        "api": "openai-completions",
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "not-needed",
        "models": [
          { "id": "auto", "name": "LLMRouter Budget Aware" }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "openclaw/auto" }
    }
  },
  "gateway": {
    "mode": "local"
  }
}
```

Then start OpenClaw normally. Requests sent to `openclaw/auto` will go to LLMRouter, and LLMRouter will pick a backend using the budget-aware config.

Quick verification before starting OpenClaw:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Write a short hello world program in Python"}
    ]
  }'
```

Notes:
- If OpenClaw and LLMRouter run in different containers or machines, replace `127.0.0.1` with the real reachable host/IP.
- The budget-aware config uses NVIDIA remote models by default.
- For local overflow capacity, uncomment the `local-overflow` block in [openclaw_router/config-budget-aware.yaml](/config/workspace/LLMRouter/openclaw_router/config-budget-aware.yaml) and point it to your local Ollama, vLLM, SGLang, or other OpenAI-compatible server.

## 7. If you want to test training

Start with a lightweight train script:

```bash
python tests/train_test/test_svmrouter.py
```

Or use the CLI directly:

```bash
python -m llmrouter.cli.router_main train \
  --router svmrouter \
  --config configs/model_config_train/svmrouter.yaml
```

## 8. Quickest query-dependent router test

`smallest_llm` and `largest_llm` ignore the question text. If you want routing to change by query, use a trained router such as `svmrouter`.

Train once:

```bash
python -m llmrouter.cli.router_main train \
  --router svmrouter \
  --config configs/model_config_train/svmrouter.yaml
```

Then test different prompts without making LLM API calls:

```bash
python -m llmrouter.cli.router_main infer \
  --router svmrouter \
  --config configs/model_config_test/svmrouter.yaml \
  --query "Solve this math word problem" \
  --route-only
```

```bash
python -m llmrouter.cli.router_main infer \
  --router svmrouter \
  --config configs/model_config_test/svmrouter.yaml \
  --query "Write Python code to reverse a linked list" \
  --route-only
```

Expected behavior:
- the router loads `saved_models/svmrouter/svmrouter.pkl`
- output may change across queries because this router uses query embeddings
- actual model responses still require `API_KEYS`; `--route-only` only shows the selected model

## 9. Check whether the router predicts models other than qwen

Use the helper script at [scripts/check_route_diversity.py](/config/workspace/LLMRouter/scripts/check_route_diversity.py) to test multiple representative queries from the bundled dataset without making any API calls.

Default usage with `svmrouter`:

```bash
python scripts/check_route_diversity.py \
  --router svmrouter \
  --config configs/model_config_test/svmrouter.yaml
```

Test `knnrouter` instead:

```bash
python scripts/check_route_diversity.py \
  --router knnrouter \
  --config configs/model_config_test/knnrouter.yaml
```

Useful flags:

```bash
python scripts/check_route_diversity.py \
  --router svmrouter \
  --config configs/model_config_test/svmrouter.yaml \
  --per-model 3 \
  --exclude-model qwen2.5-7b-instruct \
  --max-query-chars 300
```

What the script prints:
- the best-label distribution in the bundled routing dataset
- sampled queries whose expected best model is not `qwen2.5-7b-instruct`
- `expected=<dataset label>` and `predicted=<router output>` for each sample
- a final predicted-label distribution across the sampled queries

Interpretation:
- If all `predicted=` values are still `qwen2.5-7b-instruct`, your trained router is collapsing to the dominant class.
- If you see other predicted labels such as `mistral-7b-instruct-v0.3` or `llama-3.1-8b-instruct`, the router is varying by query.
- The script is route-only and does not require `API_KEYS`.

## Common issues

`ModuleNotFoundError: No module named 'llmrouter'`
- You skipped `pip install -e .`.

`ModuleNotFoundError: No module named 'yaml'`
- Dependencies are not installed yet; run `pip install -e .`.

API/auth errors during inference or chat
- Use `--route-only` for smoke tests.
- Set `API_KEYS` before calling actual LLM endpoints.
