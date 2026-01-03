# RiseLLM

A unified Python library for seamless interaction with multiple LLM providers.

## Features

- **Multi-Provider Support**: Unified interface for vLLM, OpenAI, and Google Gemini models
- **Parallel Generation**: Distributed inference across multiple GPUs using vLLM
- **Simple API**: Consistent method signatures across all providers

## Installation

```bash
pip install -e .
```

### Dependencies

```bash
pip install openai transformers google-genai vllm
```

## Quick Start

### OpenAI

```python
from risellm import OpenAILLM

llm = OpenAILLM(model_name="gpt-4")
response = llm.chat_generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response)
```

### Google Gemini

```python
from risellm import GoogleLLM

llm = GoogleLLM(model_name="gemini-2.5-flash")
response = llm.generate(
    user_prompt="Explain quantum computing",
    system_prompt="Answer concisely"
)
print(response)
```

### vLLM (Local Deployment)

```python
from risellm import VLLM

llm = VLLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    deployed_model_name="llama3.1-8b-instruct",
    base_url="localhost",
    port="8000"
)
response = llm.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the PPO algorithm"}
    ],
    chat=True
)
print(response)
```

### Parallel Generation (Multi-GPU)

```python
from risellm.generate import vllm

results = vllm.generate_vllm(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        [{"role": "user", "content": "What is AI?"}],
        [{"role": "user", "content": "What is ML?"}],
    ],
    gpus="0,1,2,3",
    responses=10,
    temperature=0.7
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI |
| `GOOGLE_API_KEY` | API key for Google Gemini |

## License

MIT
