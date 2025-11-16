

# Conversational Chatbot in Urdu

This repository contains a **Transformer-based encoder-decoder chatbot** for Urdu text. It includes the model code, trained weights, tokenizers, a Streamlit interface, and helper scripts used for training and evaluation.

---

## Repository Structure

* `chatbot_app.py` — Streamlit app to run the chatbot interactively.
* `transformer.py` — Transformer model implementation (`Transformer` class).
* `model_architecture.py` — Supporting architecture code for the model.
* `train_data.csv`, `val_data.csv`, `test_data.csv` — Dataset files (if included).
* `best_masked_model.pt` — Pre-trained model weights used by the app.
* `urdu_tokenizer.model`, `urdu_tokenizer.vocab` — SentencePiece tokenizer files.
* `NLP_Chatbot.ipynb` — Notebook for experiments and model development.
* `requirements.txt` — Python dependencies (use inside a virtual environment).

---

## Quick Start (Windows PowerShell)

1. **Create and activate a virtual environment** (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. **Install dependencies**:

```powershell
pip install -r requirements.txt
```

3. **Optional: Fix PyTorch DLL / initialization errors**:

If you run into errors (like `WinError 1114`) when importing PyTorch, install the CPU-only version:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. **Run the Streamlit app**:

```powershell
streamlit run chatbot_app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser.

---

## Running the Notebook

Open `NLP_Chatbot.ipynb` in Jupyter or VS Code. Make sure to activate the same virtual environment so all dependencies match.

---

## Model & Tokenizer

* The app expects `best_masked_model.pt` and `urdu_tokenizer.model` in the repository root.
* The model class `Transformer` is defined in `transformer.py`. Example usage:

```python
from transformer import Transformer
model = Transformer(vocab_size=sp.vocab_size())
```

---

## Troubleshooting

**Problem:** `OSError WinError 1114` when importing `torch` (DLL initialization failed).
**Cause:** Environment mismatch (Python version, GPU drivers, CUDA, or missing system libraries).

**Quick fixes:**

* **CPU-only PyTorch** (recommended if GPU not needed):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

* **GPU support**:

  1. Ensure your Python version is supported by PyTorch (commonly 3.8–3.11).
  2. Check NVIDIA driver and CUDA version (`nvidia-smi`).
  3. Reinstall PyTorch matching your CUDA driver: [PyTorch Install Guide](https://pytorch.org/get-started/locally/).
  4. Install Microsoft Visual C++ Redistributable (2015–2022).

**Check installation:**

```powershell
python -c "import torch; print(torch.__version__)"
```

If it fails, follow the environment fixes above.

---

## Notes & Tips

* Keep `urdu_tokenizer.model` and `best_masked_model.pt` next to `chatbot_app.py` so the app can locate them.
* For friendlier error messages in the Streamlit app, wrap the model loading code in a `try/except` block.

---

## Contact / Support

If you need help fixing environment issues, provide the following info:

```powershell
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi  # if you have an NVIDIA GPU
```

With this info, you can get the exact PyTorch wheel or commands for your setup.


