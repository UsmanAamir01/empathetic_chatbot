# 📦 Model Checkpoint Files

## ⚠️ Important Notice

The trained model checkpoint files (`best_model.pt` and `best_model2.pt`) are **NOT included in this repository** because they are too large (~45MB each) for GitHub's standard file size limits.

---

## 🎯 How to Get the Model Files

### Option 1: Download from GitHub Releases (Recommended)

**Coming Soon:** Model files will be available as GitHub Releases.

```bash
# Check releases page
https://github.com/UsmanAamir01/empathetic_chatbot/releases
```

### Option 2: Train Your Own Model

Follow the instructions in the notebook to train your own model:

1. Download the [Empathetic Dialogues dataset](https://github.com/facebookresearch/EmpatheticDialogues)
2. Place it in `dataset/emotion-emotion_69k.csv`
3. Open `empathetic-chatbot_imp.ipynb` in Jupyter
4. Run all cells sequentially (Tasks 1-4)
5. Model will be saved in `checkpoints/best_model.pt`

**Training time:** ~2-3 hours on GPU (NVIDIA T4/V100)

### Option 3: Contact Me

If you need the pre-trained model files, please:
- Open an issue in this repository
- Or contact me directly (see Contact section in README)

---

## 📁 Required Files for Running the App

To run the Streamlit app, you need:

### ✅ Included in Repository:
- ✅ `vocab.json` - Vocabulary file (15K tokens)
- ✅ `app.py` - Streamlit application
- ✅ `empathetic-chatbot_imp.ipynb` - Training notebook

### ❌ Need to Obtain:
- ❌ `best_model.pt` OR `best_model2.pt` - Trained model checkpoint

---

## 🚀 Quick Start (Without Model)

You can still explore the code and architecture:

```bash
# Clone the repository
git clone https://github.com/UsmanAamir01/empathetic_chatbot.git
cd empathetic_chatbot

# Install dependencies
pip install -r requirements.txt

# Explore the notebook
jupyter notebook empathetic-chatbot_imp.ipynb

# Train your own model by running all notebook cells
```

---

## 📊 Model Specifications

### best_model2.pt (Latest)
- **Epoch:** 6
- **BLEU Score:** 18.5
- **ROUGE-L:** 0.32
- **chrF:** 42.3
- **Perplexity:** 28.7
- **Size:** ~45 MB
- **Parameters:** 11.5 million

### best_model.pt (v1)
- **Epoch:** 5
- **BLEU Score:** 17.8
- **Size:** ~45 MB
- **Parameters:** 11.5 million

---

## 🔧 Model Checkpoint Contents

Each `.pt` file contains:

```python
{
    "epoch": 6,
    "model_state_dict": {...},      # Model weights (11.5M parameters)
    "optimizer_state_dict": {...},  # Optimizer state
    "vocab": {...},                 # Vocabulary mappings (15K tokens)
    "config": {
        "d_model": 256,
        "n_heads": 2,
        "num_layers": 2,
        "d_ff": 2048,
        "dropout": 0.1,
        ...
    },
    "metrics": {
        "val_loss": 3.35,
        "ppl": 28.7,
        "bleu": 18.5,
        "rougeL": 0.32,
        "chrf": 42.3
    }
}
```

---

## 💾 Using Git LFS (Alternative)

If you want to include model files in future commits:

```bash
# Install Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes

# Add model files
git add best_model2.pt
git commit -m "Add model checkpoint via Git LFS"
git push
```

**Note:** GitHub has a 2GB bandwidth limit per month for Git LFS on free accounts.

---

## 🤝 Sharing Models

If you train an improved model, consider sharing it via:
- GitHub Releases
- [Hugging Face Hub](https://huggingface.co/models)
- [Google Drive](https://drive.google.com) (share link in Issues)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## ❓ FAQ

**Q: Can I run the app without model files?**
A: No, the Streamlit app requires either `best_model.pt` or `best_model2.pt` to generate responses.

**Q: How long does training take?**
A: ~2-3 hours on a GPU (NVIDIA T4/V100). CPU training is not recommended (would take 10+ hours).

**Q: Can I use a different model architecture?**
A: Yes! The code is modular. You can modify the `TransformerModel` class in the notebook.

**Q: What if I get "checkpoint not found" error?**
A: Make sure you have either `best_model.pt` or `best_model2.pt` in the project root directory.

---

## 📞 Support

If you have issues obtaining or using the model files:
1. Check the [Issues](https://github.com/UsmanAamir01/empathetic_chatbot/issues) page
2. Open a new issue with the "model-files" label
3. Contact via email (see README)

---

**Last Updated:** October 17, 2025
