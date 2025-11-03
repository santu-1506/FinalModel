@echo off
REM Build and deploy the Python 3.8 Keras model to GCP Cloud Run

echo ========================================
echo Building Python 3.8 Model API on GCP
echo ========================================

set PROJECT_ID=meditrack-app-new

echo.
echo Step 1: Building with cloudbuild-model-api-py38.yaml...
echo.

gcloud builds submit --config cloudbuild-model-api-py38.yaml --project=%PROJECT_ID%

if errorlevel 1 (
    echo.
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ BUILD COMPLETE!
echo ========================================
echo.
echo Image: gcr.io/%PROJECT_ID%/crispr-model-api-py38:latest
echo.
echo Next, run this to deploy:
echo.
echo gcloud run deploy crispr-model-api --image gcr.io/%PROJECT_ID%/crispr-model-api-py38:latest --platform managed --region us-central1 --allow-unauthenticated --memory 4Gi --cpu 2 --timeout 300 --project=%PROJECT_ID%
echo.
echo ========================================
pause
