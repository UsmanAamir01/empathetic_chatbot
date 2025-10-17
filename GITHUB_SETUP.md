# 🚀 Push to GitHub - Step by Step Guide

## ✅ Local Repository Setup Complete!

Your git repository has been initialized and your first commit is ready.

---

## 📝 Next Steps: Create GitHub Repository and Push

### Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Configure your repository:
   - **Repository name**: `empathetic-chatbot` (or any name you prefer)
   - **Description**: "Empathetic AI Chatbot using Transformer architecture with Streamlit UI"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### Step 2: Link Your Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```powershell
# Navigate to your project directory
cd "d:\Semester 7\Gen AI\empathetic_chatbot"

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/empathetic-chatbot.git

# Verify the remote was added
git remote -v

# Rename the branch to main (if needed)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

### Step 3: Alternative - Using GitHub CLI (gh)

If you have GitHub CLI installed:

```powershell
# Navigate to your project
cd "d:\Semester 7\Gen AI\empathetic_chatbot"

# Create repository and push (it will prompt for login if needed)
gh repo create empathetic-chatbot --public --source=. --remote=origin --push
```

---

## 📋 Files Included in Your Repository

✅ **Core Files:**

- `app.py` - Streamlit web application
- `empathetic-chatbot_imp.ipynb` - Complete Jupyter notebook
- `vocab.json` - Vocabulary file (15K tokens)
- `best_model.pt` - Trained model checkpoint v1 (EXCLUDED)
- `best_model2.pt` - Trained model checkpoint v2 (EXCLUDED)

✅ **Documentation:**

- `README.md` - Comprehensive project documentation
- `LICENSE` - MIT License
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

❌ **Excluded (via .gitignore):**

- Model checkpoints (\*.pt files - too large for GitHub)
- Dataset files (_.csv, _.jsonl)
- Virtual environments
- Cache files
- Evaluation outputs

---

## 💡 Important Notes

### Model Files are Excluded

The model checkpoint files (`best_model.pt`, `best_model2.pt`) are excluded because:

- They are ~45MB each (GitHub has 100MB file size limit)
- Binary files don't work well with version control

### Solutions for Sharing Models:

**Option 1: GitHub Releases**

```powershell
# After pushing your code, create a release
gh release create v1.0.0 best_model2.pt --title "Model Checkpoint v1.0" --notes "Trained model checkpoint (BLEU: 18.5)"
```

**Option 2: Git LFS (Large File Storage)**

```powershell
# Install Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add best_model2.pt
git commit -m "Add model checkpoint via Git LFS"
git push
```

**Option 3: External Hosting**

- Upload to [Google Drive](https://drive.google.com)
- Upload to [Hugging Face Hub](https://huggingface.co)
- Upload to [Kaggle Datasets](https://www.kaggle.com/datasets)
- Add download link in README

---

## 🔐 Authentication

If GitHub asks for authentication:

### Option A: Personal Access Token (Recommended)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when prompted

### Option B: SSH Key

```powershell
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Then use SSH URL instead:
git remote add origin git@github.com:YOUR_USERNAME/empathetic-chatbot.git
```

---

## 📊 Verify Upload

After pushing, check your GitHub repository:

- ✅ All code files are present
- ✅ README displays correctly
- ✅ License is recognized
- ✅ .gitignore is working (model files not uploaded)

---

## 🎉 Success!

Your repository is now live on GitHub!

### Share your project:

```
https://github.com/YOUR_USERNAME/empathetic-chatbot
```

### Clone on another machine:

```powershell
git clone https://github.com/YOUR_USERNAME/empathetic-chatbot.git
cd empathetic-chatbot
pip install -r requirements.txt
# Download model files separately
streamlit run app.py
```

---

## 🆘 Troubleshooting

### Problem: File too large

```
Solution: Use Git LFS or exclude the file in .gitignore
```

### Problem: Authentication failed

```
Solution: Use Personal Access Token instead of password
```

### Problem: Remote already exists

```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/empathetic-chatbot.git
```

---

## 📞 Need Help?

Run these commands if something goes wrong:

```powershell
# Check git status
git status

# Check remote
git remote -v

# View commit history
git log --oneline

# Undo last commit (if needed)
git reset --soft HEAD~1
```

---

**Created on:** October 17, 2025
**Git initialized:** ✅ Complete
**First commit:** ✅ Complete
**Ready to push:** ✅ Yes
