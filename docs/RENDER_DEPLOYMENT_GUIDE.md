# 🚀 Deploying Marine Oil Spill Sentinel on Render (render.com)

This guide provides step-by-step instructions for deploying the **Marine Oil Spill Segmentation and Vessel Attribution Dashboard** on [Render](https://render.com).

---

## ⚡ Method 1: Instant Blueprint Deployment (Recommended)

Render provides Infrastructure-as-Code (IaC) via the [`render.yaml`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/render.yaml) file included in the root of this repository.

1. **Push your code to GitHub:**
   Ensure your latest code is on branch `main` at:
   `https://github.com/TalupulaYaswanth/Marine-Oil-Spill-Segmentation.git`

2. **Log in to Render:**
   Go to [https://dashboard.render.com](https://dashboard.render.com) (sign up using your GitHub account).

3. **Create a New Blueprint:**
   - Click the **"New +"** button in the top navigation bar.
   - Select **"Blueprint"**.
   - Connect your GitHub repository: `Marine-Oil-Spill-Segmentation`.
   - Render will automatically detect [`render.yaml`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/render.yaml) and populate all build commands, start commands, health checks, and environment variables.

4. **Deploy:**
   - Click **"Apply"**.
   - Render will build the environment and deploy your live Streamlit dashboard.
   - Your live public URL will be: `https://marine-oil-spill-sentinel.onrender.com` (or your custom chosen service name).

---

## 🛠️ Method 2: Manual Web Service Setup (Step-by-Step)

If you prefer to configure the Web Service manually via the Render UI:

1. **Create Web Service:**
   - On the Render Dashboard, click **"New +"** -> **"Web Service"**.
   - Select **"Build and deploy from a Git repository"**.
   - Choose `TalupulaYaswanth/Marine-Oil-Spill-Segmentation`.

2. **Configure Service Details:**
   | Field | Setting | Explanation |
   | :--- | :--- | :--- |
   | **Name** | `marine-oil-spill-sentinel` | Name of your service (defines your URL subdomain) |
   | **Region** | `Oregon (US West)` or `Frankfurt (EU)` | Choose the region closest to you |
   | **Branch** | `main` | Production deployment branch |
   | **Root Directory** | *(leave blank)* | Root of the repository |
   | **Runtime** | `Python` | Native Python runtime |
   | **Build Command** | `chmod +x build.sh && ./build.sh` | Upgrades pip, installs dependencies, pre-generates sample data |
   | **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` | Binds Streamlit to Render's dynamic `$PORT` |
   | **Instance Type** | `Free` ($0/month) | 512 MB RAM, 0.1 CPU |

3. **Configure Environment Variables:**
   Under **"Advanced"** -> **"Environment Variables"**, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `MPLCONFIGDIR`: `/tmp/matplotlib_cache`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_SERVER_ENABLE_CORS`: `false`
   - `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION`: `false`

4. **Health Check Path:**
   - Set **Health Check Path** to: `/_stcore/health`
   *(Render will ping this path to verify the Streamlit engine is live before routing traffic).*

5. **Click "Create Web Service"**:
   - Render starts building immediately. You can view the real-time build and deployment logs in the console.

---

## 🐳 Method 3: Deploying via Docker on Render

If you prefer complete containerized isolation, this repository includes an optimized [`Dockerfile`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/Dockerfile):

1. On the Render Dashboard, click **"New +"** -> **"Web Service"**.
2. Connect your repository.
3. In the **Runtime** dropdown, select **"Docker"**.
4. Render will automatically detect the [`Dockerfile`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/Dockerfile) and build the container image.
5. Set the Plan to **Free**, and click **"Create Web Service"**.

---

## 💡 Important Tips for the Render Free Tier

1. **Dynamic Port Binding (`$PORT`):**  
   Render assigns a dynamic port (e.g., 10000) stored in the environment variable `$PORT`. Streamlit **must** listen on `--server.port $PORT --server.address 0.0.0.0`. This is already configured in `render.yaml`, `build.sh`, `Dockerfile`, and `Procfile`.

2. **Spin-Down on Inactivity (Cold Starts):**  
   On the Free tier, Render puts web services to sleep after 15 minutes of inactivity. When a user navigates to your `.onrender.com` URL, it will take ~30–50 seconds to wake up and display the dashboard. Subsequent requests will be instantaneous.

3. **Memory Optimization:**  
   The Render Free tier provides 512 MB RAM. Our [`requirements.txt`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/requirements.txt) fetches the lightweight CPU wheel of PyTorch (~170 MB instead of the 2.5 GB CUDA version), and our hybrid segmentation engine automatically uses the zero-overhead **EMSA CleanSeaNet CFAR** algorithm if memory is constrained.

4. **Custom Domain:**  
   You can connect a custom domain (e.g., `oilspill.yourdomain.com`) for free in the Render dashboard under **Settings** -> **Custom Domains**.
