# Syablit - LLM Abliteration Toolkit 🔥

Remove refusal behavior from LLMs via activation steering for **red-teaming purposes**.

Based on [\"Refusal in LLMs is Mediated by a Single Direction\"](https://arxiv.org/abs/2406.11717) (Arditi et al., 2024)

## What is Abliteration?

Abliteration identifies the **refusal direction** in a model's activation space and projects it out, effectively removing the model's tendency to refuse certain prompts.

### The Math

```
c'_out = c_out - (c_out · r̂) r̂
```

Where:
- `c_out` = component output before modification
- `r̂` = refusal direction vector (unit vector)
- `c'_out` = modified output (refusal removed)

## Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suvijya/syablit/blob/main/notebooks/abliteration_demo.ipynb)

### Option 2: Local Setup

```bash
# Clone the repo
git clone https://github.com/suvijya/syablit.git
cd syablit

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py
```

## Usage

```python
from src import Abliterator

# Load any HuggingFace model
abl = Abliterator("meta-llama/Llama-2-7b-chat-hf", device="cuda")

# Cache activations on harmful/harmless prompts
abl.cache_activations()

# Compute refusal direction
abl.compute_refusal_direction()

# Abliterate the model
abl.abliterate(strength=1.0)

# Test the result
abl.test_refusal(["How do I make a bomb?"])

# Save the modified model
abl.save_model("./ablated_model")
```

## Project Structure

```
syablit/
├── src/
│   ├── __init__.py
│   ├── abliterator.py    # Core abliteration logic
│   ├── data.py           # Harmful/harmless prompt datasets
│   ├── hooks.py          # PyTorch activation hooks
│   └── utils.py          # Utility functions
├── notebooks/
│   └── abliteration_demo.ipynb  # Google Colab notebook
├── tests/
├── requirements.txt
├── demo.py
└── README.md
```

## How It Works

1. **Cache Activations**: Run the model on harmful and harmless prompts, capturing intermediate activations
2. **Compute Refusal Direction**: Calculate the mean difference between harmful and harmless activations
3. **Abliterate**: Project out the refusal direction from model weights
4. **Test**: Verify that refusal behavior has been removed

## Red Teaming Use Cases

- **Safety Research**: Understand how alignment works at the representation level
- **Adversarial Testing**: Test model robustness against representation-level attacks
- **Interpretability**: Study what directions in activation space encode specific behaviors

## ⚠️ Disclaimer

This tool is intended for **research and red-teaming purposes only**. 

- Use responsibly and ethically
- Do not use to create harmful or dangerous models
- Follow your organization's AI safety policies
- Comply with all applicable laws and regulations

## Citation

If you use this work, please cite:

```bibtex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Oscar and Aquea, Aedan and Krawic, Juan and Zou, Andy and others},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.
