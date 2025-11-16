# Conversational Chatbot in Urdu

This repository contains a Transformer encoder-decoder chatbot for Urdu text. The project includes model code, a trained model file, tokenizers, a Streamlit app interface, and helper scripts used for training and evaluation.

## Repository structure

- `chatbot_app.py` — Streamlit app to run the chatbot.
- `transformer.py` — Transformer model implementation (class `Transformer`).
- `model_architecture.py` — auxiliary model architecture code.
- `train_data.csv`, `val_data.csv`, `test_data.csv` — datasets (if present).
- `best_masked_model.pt` — trained model weights used by the app.
- `urdu_tokenizer.model`, `urdu_tokenizer.vocab` — SentencePiece tokenizer files.
- `NLP_Chatbot.ipynb` — notebook used for experiments and development.
- `requirements.txt` — Python dependencies (use inside a virtual environment).

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install project dependencies:

```powershell
pip install -r requirements.txt
```

3. If you encounter PyTorch DLL / initialization errors (see Troubleshooting), install the CPU-only PyTorch wheel as a quick workaround:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. Run the Streamlit app:

```powershell
streamlit run chatbot_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Running the notebook

Open `NLP_Chatbot.ipynb` in Jupyter or VS Code's notebook UI. Activate the same virtual environment so the notebook uses the same dependencies.

## Model / Tokenizer

- The app expects `best_masked_model.pt` and `urdu_tokenizer.model` in the repository root.
- The model class is `Transformer` defined in `transformer.py`. The Streamlit app instantiates it with:

```python
from transformer import Transformer
model = Transformer(vocab_size=sp.vocab_size())
```

## Troubleshooting

- Symptom: OSError WinError 1114 when importing `torch` (DLL initialization failed; message references `c10.dll`).
  - Cause: Native PyTorch binaries depend on system libraries, GPU drivers, and a matching Python/CUDA configuration. This is an environment issue — not a Python code bug.
  - Quick fixes:
    - If you do not need GPU acceleration, install the CPU-only PyTorch wheel:

      ```powershell
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      ```

    - If you need GPU support:
      1. Check your Python version (`python --version`). Many PyTorch Windows wheels officially support specific Python versions (commonly 3.8–3.11). If you are on Python 3.12/3.13, install a supported Python and create a virtualenv with it.
      2. Run `nvidia-smi` to check your NVIDIA driver and CUDA compatibility (if you have an NVIDIA GPU).
      3. Reinstall PyTorch matching your CUDA driver version. Use the PyTorch install selector at https://pytorch.org/get-started/locally/ to get the correct pip command.
      4. Ensure the Microsoft Visual C++ Redistributable is installed (2015-2022).

- Additional checks:
  - Run a minimal test in PowerShell to reproduce the import error:

    ```powershell
    python -c "import torch; print(torch.__version__)"
    ```

  - If the command raises the same `OSError`, follow the above environment steps.

## Notes & tips

- The Streamlit app performs model loading on startup. If you want more graceful error messages for missing or broken PyTorch installs, consider editing `chatbot_app.py` to import `torch` and load the model inside a `try/except` block and show a helpful Streamlit error message.
- Keep the `urdu_tokenizer.model` file and the `best_masked_model.pt` next to `chatbot_app.py` for the app to find them by relative path.

## Contact

If you need additional help reproducing or fixing environment issues, provide the output of:

```powershell
python --version
python -c "import torch; print(torch.__version__)"  # if it imports
nvidia-smi  # if you have an NVIDIA GPU
```

and I can suggest the exact PyTorch wheel/commands to install.

## License

Add your preferred license here (e.g., MIT) or remove this section.
