# 🚀 Deployment Guide: AI Resume Analyzer

This guide details how to deploy your AI Resume Analyzer web application so others can access it online.

---

## Option 1: Streamlit Community Cloud (Recommended & Free)

Streamlit Community Cloud is the easiest and fastest way to deploy. It connects directly to your GitHub repository and hosts your app for free.

### Step 1: Push Your Code to GitHub
1. Create a new repository on [GitHub](https://github.com) (e.g., `resume-analyzer-ai`).
2. Initialize Git in your project folder, add your files, and push them:
   ```bash
   git init
   git add .
   # Note: Your .env and chroma_db/ are ignored by default
   git commit -m "Initial commit of Resume Analyzer"
   git branch -M main
   git remote add origin https://github.com/your-username/resume-analyzer-ai.git
   git push -u origin main
   ```

### Step 2: Set Up Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **"Advanced settings..."** before deploying to add your HuggingFace token as a secret.

### Step 3: Configure Secrets (Crucial)
In the **Secrets** text box in Streamlit settings, paste your HuggingFace token using the following format:
```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_actual_token_here"
```
Streamlit will inject this secret directly into the environment variables, meaning `os.getenv("HUGGINGFACEHUB_API_TOKEN")` will work seamlessly without a `.env` file!

5. Click **"Deploy!"** 🚀 Your app will build and be live in a few minutes.

---

## Option 2: Hugging Face Spaces (Free & AI-Native)

Since this app uses Hugging Face models, hosting it directly on Hugging Face Spaces is a great alternative.

### Step 1: Create a Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set the name, license, and choose **Streamlit** as the Space SDK.
3. Select the free CPU tier (sufficient for running our lightweight embedding model).

### Step 2: Push Your Code
1. Clone the Space's Git repository or upload the files directly using the Hugging Face web UI.
2. Ensure you upload:
   - `app.py`
   - `requirements.txt`
   - The entire `src/` and `prompts/` folders.
   - *Do not upload `.env` or `chroma_db/`.*

### Step 3: Add Variables
1. Go to the **Settings** tab of your Space.
2. Scroll to **Variables and secrets**.
3. Click **"New secret"** and add:
   - **Key**: `HUGGINGFACEHUB_API_TOKEN`
   - **Value**: `hf_your_token_here`
4. Re-run or rebuild the space. It will be live!

---

## Option 3: Dockerized Deployment (Any Cloud Provider)

If you want to host on AWS, Google Cloud, DigitalOcean, Render, or Railway, you can containerize the application using Docker.

### Step 1: Create a `Dockerfile`
Create a file named `Dockerfile` in the root of your project:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY src/ ./src/
COPY prompts/ ./prompts/

# Expose Streamlit port
EXPOSE 8501

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Run Locally
```bash
# Build the Docker image
docker build -t resume-analyzer .

# Run the Docker container (passing HF token as env variable)
docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN="your_token" resume-analyzer
```

### Step 3: Deploy to Render / Railway
1. Push your code (including the `Dockerfile`) to GitHub.
2. Connect your repo to [Render](https://render.com) or [Railway](https://railway.app) as a Web Service.
3. Add `HUGGINGFACEHUB_API_TOKEN` in the environment variable settings.
4. The platform will auto-detect the `Dockerfile`, build it, and launch your application!
