# 🚀 Deploy Model to GitHub Release - Instructions

## 📦 Model File Sizes
- `best_model.pt`: 136.92 MB
- `best_model2.pt`: 171.1 MB

Both files exceed GitHub's 100MB limit, so we'll use **GitHub Releases**.

---

## 🎯 Step 1: Create a GitHub Release with Model Files

### Option A: Using GitHub Web Interface (Easiest)

1. **Go to your repository**: https://github.com/UsmanAamir01/empathetic_chatbot

2. **Click on "Releases"** (right sidebar)

3. **Click "Create a new release"**

4. **Fill in the release details**:
   - **Tag version**: `v1.0.0`
   - **Release title**: `Model Checkpoint v1.0 - Empathetic Chatbot`
   - **Description**:
     ```
     Pre-trained Transformer model for empathetic response generation.
     
     **Model Specifications:**
     - Architecture: Transformer Encoder-Decoder (from scratch)
     - Parameters: 11.5 million
     - Training: Empathetic Dialogues dataset (69K conversations)
     - BLEU Score: 18.5
     - ROUGE-L: 0.32
     - chrF: 42.3
     
     **Files:**
     - `best_model2.pt` (171 MB) - Latest model (epoch 6)
     - `best_model.pt` (137 MB) - First version (epoch 5)
     
     **Usage:**
     Download `best_model2.pt` and place it in the project root directory before running the Streamlit app.
     ```

5. **Attach files**:
   - Drag and drop `best_model2.pt` (primary model)
   - Drag and drop `best_model.pt` (backup)

6. **Click "Publish release"**

7. **Copy the download URL**: After publishing, right-click on `best_model2.pt` → Copy link address
   - It will look like: `https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt`

### Option B: Using GitHub CLI (If Installed)

```powershell
# Navigate to project
cd "d:\Semester 7\Gen AI\empathetic_chatbot"

# Create release and upload files
gh release create v1.0.0 `
  best_model2.pt `
  best_model.pt `
  --title "Model Checkpoint v1.0 - Empathetic Chatbot" `
  --notes "Pre-trained Transformer model for empathetic response generation. See README for details."

# Get the download URL
gh release view v1.0.0 --json assets --jq '.assets[0].url'
```

---

## 🎯 Step 2: Update app.py with Auto-Download

After creating the release, I'll update `app.py` to automatically download the model if it's not present.

**The download URL will be:**
```
https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt
```

---

## 🎯 Step 3: Deploy to Streamlit Cloud

### Prerequisites
1. ✅ GitHub repository: https://github.com/UsmanAamir01/empathetic_chatbot
2. ✅ Model uploaded to GitHub Releases
3. ✅ app.py updated with auto-download code
4. ⚠️ Streamlit Cloud account (free): https://streamlit.io/cloud

### Deployment Steps

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure the app**:
   - **Repository**: `UsmanAamir01/empathetic_chatbot`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (e.g., `empathetic-chatbot-usman`)

5. **Advanced settings** (optional):
   - **Python version**: 3.10 or 3.11
   - **Secrets**: Not needed (model URL is public)

6. **Click "Deploy!"**

7. **Wait for deployment** (~5-10 minutes):
   - First run will download the model (171 MB)
   - Subsequent runs will use cached model

8. **Your app will be live at**:
   ```
   https://empathetic-chatbot-usman.streamlit.app
   ```

---

## 📋 Checklist

Before deploying, make sure:

- [ ] Model uploaded to GitHub Release (v1.0.0)
- [ ] Download URL copied
- [ ] app.py updated with auto-download code
- [ ] requirements.txt is up-to-date
- [ ] All changes committed and pushed
- [ ] Streamlit Cloud account created
- [ ] App deployed and tested

---

## 🐛 Troubleshooting

### Issue: "Model download failed"
**Solution**: Check the release URL is correct and public

### Issue: "Out of memory" on Streamlit Cloud
**Solution**: Streamlit Cloud free tier has 1GB RAM. The model should fit, but if not:
- Use `best_model.pt` (smaller)
- Or upgrade to paid tier

### Issue: "App takes too long to load"
**Solution**: First load downloads model (~2 minutes). Subsequent loads are fast (<5 seconds)

### Issue: "Module not found"
**Solution**: Check all dependencies are in `requirements.txt`

---

## 📞 Next Steps

1. **Create the GitHub Release** (Option A or B above)
2. **Let me know when done**, and I'll update app.py with the download URL
3. **Deploy to Streamlit Cloud**
4. **Share your live app link!** 🎉

---

**Estimated Total Time**: 15-20 minutes
