# Adding New Models

This guide explains how to integrate a new language model into the evaluation framework.

---

## Option 1: Use a Supported Provider

If your model is hosted on OpenAI or Anthropic, simply pass the model string in your config or CLI:

```yaml
# configs/default.yaml
models:
  - name: gpt-4-turbo
    provider: openai
```

```bash
python experiments/run_single_model.py --model gpt-4-turbo --provider openai
```

Supported providers: `openai`, `anthropic`, `huggingface`.

---

## Option 2: Implement a Custom Client

Subclass `BaseModelClient` from `src/utils/api_client.py`:

```python
from utils.api_client import BaseModelClient

class MyCustomClient(BaseModelClient):

    def __init__(self, model: str):
        self._model = model
        # Initialize your client here

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict],
        system: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> str:
        # Call your model API and return the response as a plain string
        response = my_api.call(messages=messages, ...)
        return response.text
```

Then register it in the `build_client` factory:

```python
# src/utils/api_client.py

def build_client(model: str, provider: str) -> BaseModelClient:
    ...
    elif provider == "my_provider":
        return MyCustomClient(model=model)
```

---

## Option 3: HuggingFace Local Model

Any HuggingFace model with a chat template can be used:

```yaml
models:
  - name: meta-llama/Llama-3-8B-Instruct
    provider: huggingface
```

Requires:
```bash
pip install transformers torch accelerate
export HF_TOKEN=hf_...
```

---

## Tips for Model Comparisons

The paper compares four alignment tiers:

| Tier | Example Models |
|------|---------------|
| Base (no alignment) | GPT-2, LLaMA base |
| Instruction-tuned | Llama-3-8B-Instruct |
| RLHF-aligned | GPT-3.5-turbo, Claude Haiku |
| Preference-optimized (DPO) | Zephyr, Tulu-2-DPO |

For meaningful comparisons, keep `temperature`, `top_p`, and `max_tokens` identical across models.
