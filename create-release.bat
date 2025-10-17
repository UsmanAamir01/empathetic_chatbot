@echo off
echo ================================
echo GitHub Release Creator (Simple)
echo ================================
echo.

cd /d "d:\Semester 7\Gen AI\empathetic_chatbot"

echo Checking files...
if not exist "best_model2.pt" (
    echo ERROR: best_model2.pt not found!
    pause
    exit /b 1
)

echo Found: best_model2.pt
echo.

echo Creating GitHub Release v1.0.0...
echo This will take 5-10 minutes to upload 171 MB
echo.

gh release create v1.0.0 best_model2.pt best_model.pt --title "Model Checkpoint v1.0" --notes "Pre-trained Empathetic Chatbot Model (BLEU: 18.5, 11.5M params). Download best_model2.pt to run the app."

if %errorlevel% equ 0 (
    echo.
    echo ================================
    echo SUCCESS! Release created!
    echo ================================
    echo.
    echo Release URL: https://github.com/UsmanAamir01/empathetic_chatbot/releases/tag/v1.0.0
    echo Download URL: https://github.com/UsmanAamir01/empathetic_chatbot/releases/download/v1.0.0/best_model2.pt
    echo.
    echo Next: Run "streamlit run app.py" to test
    echo.
) else (
    echo.
    echo ERROR: Failed to create release
    echo.
    echo Try manual method:
    echo 1. Open: https://github.com/UsmanAamir01/empathetic_chatbot/releases/new
    echo 2. Tag: v1.0.0
    echo 3. Upload: best_model2.pt
    echo 4. Publish
)

pause
