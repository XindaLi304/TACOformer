# TACOformer

Official PyTorch implementation of **Token–Channel Compounded Cross-Attention** (TACO) for multimodal emotion recognition.

---

## 📦 Repository Structure

tacoformer/
├─ main.py # Entry: load data → split & save test set → k-fold hyperparameter search
│ # → train best model on training split → evaluate on held-out test set
├─ config.py # Paths and hyperparameter grid configuration
├─ data.py # Data loading, concatenation, splitting, and DataLoader helpers
├─ model.py # ViT + TACO cross-attention model (original functionality, English comments)
├─ train.py # Training and evaluation loops (keeps your loss/accuracy logic)
├─ search.py # k-fold cross-validation hyperparameter search
├─ utils.py # Utilities (seeding, device print helpers, etc.)
└─ requirements.txt # Python dependencies

---

## 🧪 Data & Preprocessing

- The processed array (merged EEG/EOG/EMG) should have the shape:
[1280, 60, n_channels, 128]
where:
- `1280` = number of trials,
- `60`   = timesteps per trial,
- `n_channels` ： EEG：81 ，EOG：4 ， EMG：4 
- `128`  = segment length per timestep.

- Preprocessing details follow the paper:  
**TACO** — *Token–Channel Compounded Cross-Attention for Multimodal Emotion Recognition*  
PDF: https://arxiv.org/pdf/2306.13592

> **Note:** This repo expects separate `.npy` files for EEG, EOG, EMG, and labels, then concatenates modalities along the channel dimension. Paths are configured in `config.py`.

---

## 🚀 Quick Start

```bash
# Clone and enter the project
git clone <your-repo-url>.git
cd tacoformer

# (Optional) Create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

If you use this code, please cite the TACO paper:

Token–Channel Compounded Cross-Attention for Multimodal Emotion Recognition, 2023.
arXiv:2306.13592
