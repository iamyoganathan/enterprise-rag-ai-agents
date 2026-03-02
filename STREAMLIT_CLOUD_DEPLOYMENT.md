# 🚀 Streamlit Cloud Deployment Guide

## Overview

This guide shows you how to deploy your Enterprise RAG System to **Streamlit Cloud (FREE)** for the frontend and **Render.com (FREE)** for the backend API.

**Total Cost: $0/month** (on free tiers)

---

## Architecture

```
┌─────────────────────────────────┐
│   Streamlit Cloud (Frontend)    │  ← FREE
│   https://yourapp.streamlit.app │
└─────────────┬───────────────────┘
              │ API Calls
              ▼
┌─────────────────────────────────┐
│   Render.com (Backend API)      │  ← FREE
│   https://yourapp.onrender.com  │
└─────────────────────────────────┘
```

---

## Prerequisites

- [x] GitHub account
- [x] Streamlit Cloud account (free - uses GitHub login)
- [x] Render.com account (free)
- [x] Your code pushed to GitHub

---

## STEP 1: Push Code to GitHub (5 minutes)

### 1.1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `enterprise-rag-ai-agents`
3. Make it **Public** (for free Streamlit Cloud)
4. Don't initialize with README (you already have one)
5. Click **Create repository**

### 1.2: Push Your Code

```powershell
cd d:\GenAI\enterprise-rag-ai-agents

# Initialize git (if not already)
git init

# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/enterprise-rag-ai-agents.git

# Add all files
git add .

# Commit
git commit -m "Initial commit - Enterprise RAG System"

# Push to GitHub
git branch -M main
git push -u origin main
```

**⚠️ IMPORTANT**: Make sure `.env` files are in `.gitignore` (don't push API keys!)

---

## STEP 2: Deploy Backend to Render.com (10 minutes)

### 2.1: Create Render Account

1. Go to https://render.com/
2. Click **"Get Started"**
3. Sign up with GitHub (easiest)
4. Verify your email

### 2.2: Create Web Service

1. In Render dashboard, click **"New +"** → **"Web Service"**
2. **Connect your GitHub repository**:
   - Click **"Connect account"** → Authorize Render
   - Select `enterprise-rag-ai-agents` repo
3. **Configure the service**:
   ```
   Name: rag-backend
   Region: Oregon (or closest to you)
   Branch: main
   Root Directory: (leave blank)
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
   ```
4. **Select plan**: **Free** (512MB RAM, sleeps after 15min inactivity)

### 2.3: Add Environment Variables

In the **Environment** section, add these:

```
GROQ_API_KEY = your_groq_api_key_here
SECRET_KEY = your_random_secret_key
DEFAULT_LLM_PROVIDER = groq
DEFAULT_LLM_MODEL = llama-3.3-70b-versatile
VECTOR_DB_PATH = /opt/render/project/data/vector_db
LOG_LEVEL = INFO
DEBUG = false
RATE_LIMIT_PER_MINUTE = 30
RATE_LIMIT_PER_HOUR = 500
```

**Generate SECRET_KEY:**
```powershell
# On Windows PowerShell:
-join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
```

### 2.4: Deploy

1. Click **"Create Web Service"**
2. Wait 5-10 minutes for first deployment
3. You'll get a URL like: `https://rag-backend-xyz.onrender.com`

### 2.5: Test Backend

```powershell
# Test health endpoint (replace with your Render URL)
curl https://rag-backend-xyz.onrender.com/health
```

**Copy your backend URL** - you'll need it for Streamlit!

---

## STEP 3: Deploy Frontend to Streamlit Cloud (5 minutes)

### 3.1: Create Streamlit Cloud Account

1. Go to https://share.streamlit.io/
2. Click **"Sign up"**
3. Sign in with GitHub
4. Authorize Streamlit Cloud

### 3.2: Deploy App

1. Click **"New app"**
2. **Select your repository**: `enterprise-rag-ai-agents`
3. **Configure**:
   ```
   Branch: main
   Main file path: frontend/app.py
   App URL: (custom name, e.g., my-rag-app)
   ```
4. Click **"Advanced settings"**

### 3.3: Add Secrets

In the **Secrets** section, paste:

```toml
# Add your backend URL from Render
API_BASE_URL = "https://rag-backend-xyz.onrender.com"

# Optional: If frontend needs direct API access
GROQ_API_KEY = "your_groq_key_here"
```

**Replace `rag-backend-xyz.onrender.com` with your actual Render URL!**

### 3.4: Deploy

1. Click **"Deploy!"**
2. Wait 2-3 minutes
3. Your app will be live at: `https://my-rag-app.streamlit.app`

---

## STEP 4: Test Your Deployment

### 4.1: Open Your App

Go to: `https://your-app-name.streamlit.app`

### 4.2: Verify Connection

- Frontend should load ✅
- Check sidebar - should show backend URL ✅
- Try asking a question ✅
- Upload a document ✅

### 4.3: Check Backend Direct

```
https://rag-backend-xyz.onrender.com/docs
```

You should see the API documentation.

---

## Free Tier Limitations

### Render.com Free Tier:
- ✅ 512MB RAM
- ✅ Shared CPU
- ⚠️ **Sleeps after 15 minutes of inactivity** (wakes up in 30 seconds)
- ✅ 750 hours/month (plenty for testing/demos)
- ✅ Auto-deploy on git push

### Streamlit Cloud Free Tier:
- ✅ Unlimited apps (public)
- ✅ 1GB RAM per app
- ✅ Community support
- ⚠️ App sleeps if inactive (wakes instantly)
- ✅ Custom subdomain
- ❌ Private apps require paid plan ($20/mo)

---

## Upgrading (If Needed)

### When to Upgrade:

**Render.com Starter ($7/mo)**:
- 512MB RAM
- Doesn't sleep
- Always-on
- Good for production demos

**Streamlit Cloud Team ($20/mo)**:
- Private apps
- Password protection
- Priority support

---

## Common Issues & Solutions

### 1. Backend Not Responding

**Problem**: Frontend shows "Backend unreachable"

**Solution**: 
- Render free tier sleeps after 15min
- First request takes 30 seconds to wake up
- Wait and try again
- Or upgrade to Starter plan ($7/mo)

### 2. API Key Missing

**Problem**: "GROQ_API_KEY not found"

**Solution**:
- Go to Render dashboard → Your service → Environment
- Add `GROQ_API_KEY` variable
- Redeploy

### 3. Frontend Can't Connect to Backend

**Problem**: CORS errors or 404

**Solution**:
- Check `API_BASE_URL` in Streamlit secrets
- Make sure it's `https://rag-backend-xyz.onrender.com` (no trailing slash)
- Verify backend is running (check Render logs)

### 4. Out of Memory

**Problem**: Backend crashes with memory errors

**Solution**:
- Reduce chunk size in settings
- Limit number of documents
- Upgrade to paid plan (2GB RAM for $25/mo)

---

## Monitoring & Logs

### Render Logs:
1. Go to Render dashboard
2. Click your service
3. Click **"Logs"** tab
4. View real-time logs

### Streamlit Logs:
1. Go to https://share.streamlit.io/
2. Click your app
3. Click **"Manage app"** → **"Logs"**

---

## Auto-Deploy on Git Push

Both services auto-deploy when you push to GitHub:

```powershell
cd d:\GenAI\enterprise-rag-ai-agents

# Make changes
# ...

# Commit and push
git add .
git commit -m "Updated feature"
git push

# Render and Streamlit will auto-deploy in 2-5 minutes
```

---

## Cost Comparison

| Setup | Frontend | Backend | Monthly Cost |
|-------|----------|---------|--------------|
| **Free Tier** | Streamlit Cloud | Render Free | $0 |
| **Always-On** | Streamlit Cloud | Render Starter | $7 |
| **Private** | Streamlit Team | Render Starter | $27 |
| **Production** | Streamlit Team | Render Pro | $45 |

---

## Alternative: Railway.app (Backend)

Instead of Render, you can use Railway.app:

1. Go to https://railway.app/
2. Sign up with GitHub
3. **New Project** → **Deploy from GitHub**
4. Select your repo
5. Add environment variables
6. Railway auto-detects Python and runs it
7. Get URL like: `https://rag-backend.up.railway.app`

**Railway Free Tier**: $5 free credits/month (enough for ~500 hours)

---

## Production Checklist

Before going live:

- [ ] All API keys set in environment variables
- [ ] SECRET_KEY changed from default
- [ ] CORS configured with actual domain
- [ ] Rate limiting enabled
- [ ] Error monitoring set up (Sentry)
- [ ] Backup strategy for vector database
- [ ] Custom domain configured (optional)
- [ ] SSL certificate (auto with Render/Railway)
- [ ] Load testing completed
- [ ] Documentation updated with new URLs

---

## Support

**Render.com**: https://render.com/docs  
**Streamlit Cloud**: https://docs.streamlit.io/streamlit-community-cloud  
**Railway.app**: https://docs.railway.app/

**Your deployed app will be at**:
- Frontend: `https://your-app.streamlit.app`
- Backend API: `https://your-backend.onrender.com`
- API Docs: `https://your-backend.onrender.com/docs`

---

## Summary

✅ **Total Time**: 20-30 minutes  
✅ **Total Cost**: $0/month (free tier)  
✅ **Maintenance**: Auto-deploy on git push  
✅ **Scalability**: Upgrade when needed  

This is the **simplest production deployment** for RAG systems! 🎉
