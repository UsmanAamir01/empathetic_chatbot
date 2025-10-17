# 🚀 Creating GitHub Release - Command Line Guide

## ⚡ Quick Start (Choose ONE method)

### Method 1: PowerShell Script (Recommended)

```powershell
# Close and reopen PowerShell terminal first (to load gh command)
# Then run:
cd "d:\Semester 7\Gen AI\empathetic_chatbot"
.\create-release.ps1
```

### Method 2: Batch Script (Alternative)

```cmd
# Double-click: create-release.bat
# OR run in Command Prompt:
cd "d:\Semester 7\Gen AI\empathetic_chatbot"
create-release.bat
```

### Method 3: Manual Commands (Step by Step)

**IMPORTANT: Close and reopen your PowerShell terminal first!**

Then run these commands one by one:

```powershell
# 1. Navigate to project
cd "d:\Semester 7\Gen AI\empathetic_chatbot"

# 2. Verify gh is installed
gh --version

# 3. Login to GitHub (if not already)
gh auth login
# Select: GitHub.com → HTTPS → Login with browser → Authorize

# 4. Create the release
gh release create v1.0.0 best_model2.pt best_model.pt `
  --title "Model Checkpoint v1.0 - Empathetic Chatbot" `
  --notes "Pre-trained model: BLEU 18.5, 11.5M params, 171MB. Download to run the app."

# 5. Verify release was created
gh release view v1.0.0
```

---

## 📋 Step-by-Step Walkthrough

### Step 1: Restart PowerShell Terminal

**Why?** GitHub CLI was just installed and the terminal needs to reload PATH.

1. **Close** your current PowerShell terminal
2. **Open** a new PowerShell terminal
3. **Navigate** to project:
   ```powershell
   cd "d:\Semester 7\Gen AI\empathetic_chatbot"
   ```

### Step 2: Verify Installation

```powershell
gh --version
```

**Expected output:**
```
gh version 2.81.0 (2024-XX-XX)
```

**If you see "command not found"**: Restart PowerShell again!

### Step 3: Authenticate with GitHub

```powershell
gh auth login
```

**Follow the prompts:**
1. Select: `GitHub.com`
2. Protocol: `HTTPS`
3. Authenticate: `Login with a web browser`
4. Copy the code shown
5. Press Enter (opens browser)
6. Paste code and authorize

**Or check if already logged in:**
```powershell
gh auth status
```

### Step 4: Verify Model Files

```powershell
Get-ChildItem *.pt | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
```

**Expected output:**
```
Name           Size(MB)
----           --------
best_model.pt    136.92
best_model2.pt   171.1
```

### Step 5: Create the Release

```powershell
gh release create v1.0.0 best_model2.pt best_model.pt `
  --title "Model Checkpoint v1.0 - Empathetic Chatbot" `
  --notes "Pre-trained Transformer model (BLEU: 18.5, ROUGE-L: 0.32, chrF: 42.3). Download best_model2.pt to run the Streamlit app. See README for full details."
```

**What happens:**
- Creates tag `v1.0.0`
- Uploads `best_model2.pt` (171 MB) - takes ~5-10 minutes
- Uploads `best_model.pt` (137 MB) - takes ~3-5 minutes
- Creates release with description

**Progress output:**
```
✓ Creating release v1.0.0
✓ Uploading best_model2.pt [████████████] 100%
✓ Uploading best_model.pt [████████████] 100%
✓ https://github.com/UsmanAamir01/empathetic_chatbot/releases/tag/v1.0.0
```

### Step 6: Verify Release

```powershell
gh release view v1.0.0
```

**Or visit in browser:**
```
https://github.com/UsmanAamir01/empathetic_chatbot/releases
```

---

## ✅ Testing the Release

### Test 1: Check Download URL

```powershell
# Get download URL
gh release view v1.0.0 --json assets --jq '.assets[0].browser_download_url'
```

**Expected output:**
```
https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt
```

### Test 2: Download Model (Verify)

```powershell
# Test download (to a test file)
Invoke-WebRequest -Uri "https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt" -OutFile "test_download.pt"

# Check size
Get-ChildItem test_download.pt | Select-Object Name, Length

# Clean up test file
Remove-Item test_download.pt
```

### Test 3: Run Streamlit App

```powershell
# Delete local model to test auto-download
Remove-Item best_model2.pt

# Run app (should auto-download from release)
streamlit run app.py
```

**Expected behavior:**
1. App detects missing model
2. Shows download progress bar
3. Downloads from GitHub Release
4. Loads model
5. App ready to chat!

---

## 🐛 Troubleshooting

### Error: "gh: command not found"

**Solution:**
```powershell
# 1. Close PowerShell
# 2. Open NEW PowerShell
# 3. Try again: gh --version
```

### Error: "Not authenticated"

**Solution:**
```powershell
gh auth login
# Follow browser authentication
```

### Error: "Release already exists"

**Solution:**
```powershell
# Delete existing release first
gh release delete v1.0.0 --yes

# Then create again
gh release create v1.0.0 best_model2.pt ...
```

### Error: "File not found"

**Solution:**
```powershell
# Make sure you're in the right directory
cd "d:\Semester 7\Gen AI\empathetic_chatbot"

# List files
Get-ChildItem *.pt
```

### Error: "Upload timeout" or "Connection lost"

**Solution:**
```powershell
# Upload one file at a time
gh release create v1.0.0 best_model2.pt --title "Model v1.0" --notes "Model"

# Add second file later (if needed)
gh release upload v1.0.0 best_model.pt
```

### Error: "Permission denied"

**Solution:**
```powershell
# Make sure you have push access to the repo
gh auth status

# Check repo access
gh repo view UsmanAamir01/empathetic_chatbot
```

---

## 📊 Progress Indicators

While uploading, you'll see:

```
✓ Creating release v1.0.0
  Uploading best_model2.pt
  [████████░░░░░░░░░░░░] 45% | 77 MB / 171 MB | 2.5 MB/s | ETA: 37s
```

**Upload times (approximate):**
- 10 Mbps connection: ~15-20 minutes
- 50 Mbps connection: ~5-8 minutes
- 100 Mbps connection: ~3-5 minutes

---

## 🎯 After Success

Once release is created:

```powershell
# 1. Update todo list
Write-Host "✓ Release created!" -ForegroundColor Green

# 2. Test Streamlit locally
streamlit run app.py

# 3. Deploy to Streamlit Cloud
# Go to: https://share.streamlit.io/
```

---

## 🔗 Useful Commands

```powershell
# List all releases
gh release list

# View specific release
gh release view v1.0.0

# Delete release
gh release delete v1.0.0 --yes

# Upload additional file
gh release upload v1.0.0 newfile.pt

# Download release asset
gh release download v1.0.0 --pattern "*.pt"

# Edit release notes
gh release edit v1.0.0 --notes "New description"
```

---

## 📞 Quick Reference

| Command | Description |
|---------|-------------|
| `gh --version` | Check installation |
| `gh auth login` | Login to GitHub |
| `gh auth status` | Check login status |
| `gh release create` | Create new release |
| `gh release view` | View release details |
| `gh release list` | List all releases |

---

## ✨ Next Steps After Release

1. ✅ Mark todo as complete
2. ✅ Test download URL
3. ✅ Run Streamlit locally
4. ✅ Deploy to Streamlit Cloud
5. ✅ Update README with live link

---

**Total time: 10-15 minutes** (including upload time)

**Let me know when the release is created!** 🚀
