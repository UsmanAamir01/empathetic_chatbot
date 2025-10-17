# GitHub Release Creation Script
# Run this after restarting PowerShell terminal

Write-Host "================================" -ForegroundColor Cyan
Write-Host "GitHub Release Creator" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location "d:\Semester 7\Gen AI\empathetic_chatbot"

Write-Host "Step 1: Checking GitHub CLI..." -ForegroundColor Yellow
try {
    $ghVersion = gh --version
    Write-Host "✓ GitHub CLI installed: $($ghVersion[0])" -ForegroundColor Green
} catch {
    Write-Host "✗ GitHub CLI not found. Please restart PowerShell terminal." -ForegroundColor Red
    Write-Host "  After restart, run: gh --version" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 2: Checking authentication..." -ForegroundColor Yellow
try {
    $authStatus = gh auth status 2>&1
    if ($authStatus -match "Logged in") {
        Write-Host "✓ Already authenticated with GitHub" -ForegroundColor Green
    } else {
        throw "Not authenticated"
    }
} catch {
    Write-Host "⚠ Need to authenticate with GitHub" -ForegroundColor Yellow
    Write-Host "  Running: gh auth login" -ForegroundColor Cyan
    gh auth login
}

Write-Host ""
Write-Host "Step 3: Checking model files..." -ForegroundColor Yellow
$model1 = "best_model2.pt"
$model2 = "best_model.pt"

if (Test-Path $model1) {
    $size1 = [math]::Round((Get-Item $model1).Length / 1MB, 2)
    Write-Host "✓ Found $model1 ($size1 MB)" -ForegroundColor Green
} else {
    Write-Host "✗ Missing $model1" -ForegroundColor Red
    exit 1
}

if (Test-Path $model2) {
    $size2 = [math]::Round((Get-Item $model2).Length / 1MB, 2)
    Write-Host "✓ Found $model2 ($size2 MB)" -ForegroundColor Green
} else {
    Write-Host "⚠ Missing $model2 (optional backup)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 4: Creating GitHub Release v1.0.0..." -ForegroundColor Yellow
Write-Host "  This will upload $size1 MB + $size2 MB (may take 5-10 minutes)" -ForegroundColor Cyan

$releaseNotes = @"
## 🤖 Pre-trained Empathetic Chatbot Model

**Model Specifications:**
- Architecture: Transformer Encoder-Decoder (built from scratch)
- Parameters: 11.5 million
- Training Dataset: Empathetic Dialogues (69K conversations)
- Training Epochs: 6

**Performance Metrics:**
- BLEU Score: 18.5
- ROUGE-L: 0.32
- chrF: 42.3
- Perplexity: 28.7

**Files Included:**
- ``best_model2.pt`` (171 MB) - Latest model checkpoint (Recommended)
- ``best_model.pt`` (137 MB) - Earlier checkpoint (Backup)

**Usage:**
- For Streamlit Cloud: Model auto-downloads on first run
- For local use: Download ``best_model2.pt`` and place in project root

**Download and run:**
``````bash
wget https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt
streamlit run app.py
``````
"@

try {
    Write-Host ""
    Write-Host "Creating release..." -ForegroundColor Cyan
    
    if (Test-Path $model2) {
        gh release create v1.0.0 $model1 $model2 `
            --title "Model Checkpoint v1.0 - Empathetic Chatbot" `
            --notes $releaseNotes
    } else {
        gh release create v1.0.0 $model1 `
            --title "Model Checkpoint v1.0 - Empathetic Chatbot" `
            --notes $releaseNotes
    }
    
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS! Release created!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Release URL: https://github.com/UsmanAamir01/empathetic_chatbot/releases/tag/v1.0.0" -ForegroundColor Cyan
    Write-Host "Download URL: https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Test download: Invoke-WebRequest -Uri 'https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt' -OutFile 'test.pt'" -ForegroundColor White
    Write-Host "2. Run Streamlit: streamlit run app.py" -ForegroundColor White
    Write-Host "3. Deploy to Streamlit Cloud: https://share.streamlit.io/" -ForegroundColor White
    
} catch {
    Write-Host ""
    Write-Host "✗ Failed to create release" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual alternative:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://github.com/UsmanAamir01/empathetic_chatbot/releases/new" -ForegroundColor White
    Write-Host "2. Tag: v1.0.0" -ForegroundColor White
    Write-Host "3. Upload: best_model2.pt" -ForegroundColor White
    Write-Host "4. Click: Publish release" -ForegroundColor White
}
