# FinShield AI — permanent public deployment

This folder is a **self-contained cloud build** of your dashboard. It is 26 MB
instead of 5.8 GB, needs no PyTorch, no `shap` and no `lightgbm`, boots in about
3 seconds and uses ~286 MB of RAM — which is what makes it fit on a free host.

Every number it shows is still the real number from your notebooks. Nothing was
estimated or rounded to make it smaller. See *What changed* at the bottom.

---

## IMPORTANT — deploy THIS folder, not the parent

The parent `FinShieldAI` repo (`tjain2004/FINSHEILD-AI`) will **not** work on
Render. It has no `results/` folder — those CSVs are far over GitHub's 100 MB
file limit and were never pushed — and its `requirements.txt` pins numpy 1.24,
which cannot unpickle models saved under numpy 2.x. A service built from it
reports `csvs_found: 0` and `xgboost_loaded: false`.

Everything Render needs is in **this** folder, and only this folder.

---

## Part 1 — Put the code on GitHub (about 5 minutes)

Render can only deploy from a Git repository, so this step is unavoidable.

### 1.1 Make an empty repository

You already have a GitHub account (`tjain2004`), so there is nothing to sign up
for. You do need a **new, separate** repo — the existing `FINSHEILD-AI` one has
gigabyte-sized files in its history and cannot take this.

1. Click the **+** in the top-right → **New repository**
2. **Repository name:** `finshield-ai`
3. Leave it **Public** (Render's free tier can't read private repos)
4. Do **not** tick "Add a README" — leave every checkbox empty
5. Click **Create repository**

You'll land on a page showing setup commands. Keep this tab open.

### 1.2 Upload this folder

**Easiest:** double-click **`PUSH_TO_GITHUB.bat`** in this folder and press Enter
when it asks for the repository URL. It does all of the below for you.

To do it by hand instead, open **Command Prompt** (not PowerShell — it has no
`&&`) and run these one line at a time:

```cmd
cd C:\Users\tjain\Downloads\FinShieldAI\deploy
git init
git add .
git commit -m "FinShield AI dashboard"
git branch -M main
git remote add origin https://github.com/tjain2004/finshield-ai.git
git push -u origin main
```

A browser window will pop up asking you to sign in to GitHub. Sign in, allow it,
and the push will finish. Refresh your GitHub page — all the files should be there.

> **If `git` is not recognised:** install it from https://git-scm.com/download/win,
> click Next through every screen, close Command Prompt, open a new one, and
> run the commands again.

---

## Part 2 — Deploy on Render (about 5 minutes)

### 2.1 Sign up

Go to **https://render.com** → **Get Started** → **GitHub**. Authorise Render to
read your repositories. No credit card is asked for.

### 2.2 Create the web service

1. On the Render dashboard click **+ New** → **Web Service**
2. Find `finshield-ai` in the repository list → **Connect**
3. Render reads `render.yaml` and fills most of it in. Check these:

   | Field | Value |
   |---|---|
   | **Name** | `finshield-ai` (this becomes your URL) |
   | **Language** | Python 3 |
   | **Branch** | `main` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 --preload` |
   | **Root Directory** | *(leave blank — this folder IS the repo root)* |
   | **Instance Type** | **Free** |

4. Click **Deploy Web Service**

> If you already made a service from the wrong repo, do not try to repoint it.
> Create this one fresh, confirm it works, then delete the old one.

### 2.3 Wait

The first build takes **5–10 minutes** — it is downloading pandas, scikit-learn,
XGBoost. Watch the log. You are done when you see:

```
==> Your service is live 🎉
```

### 2.4 Your permanent link

At the top of the page:

```
https://finshield-ai.onrender.com
```

**That is it.** That link works forever, from any device, with your laptop shut.
Put it in your report. It never changes unless you rename the service.

---

## Part 3 — The one thing to know about the free tier

Render puts a free service to sleep after **15 minutes with no visitors**. The
next person to open the link wakes it up, which takes **about 50 seconds** —
they'll see a blank page or a spinner, then the dashboard loads normally and
stays fast.

**This matters in a viva.** Two minutes before you present, open the link
yourself. That wakes it up, and it stays awake as long as anyone keeps using it.

If you want it to never sleep, Render's paid Starter tier is $7/month. You don't
need it — just wake it before you demo.

---

## Updating it later

Any time you change something:

```cmd
cd C:\Users\tjain\Downloads\FinShieldAI\deploy
git add .
git commit -m "what I changed"
git push
```

Render sees the push and redeploys automatically in a few minutes. Same URL.

---

## What changed vs your local dashboard

Three tabs used to read enormous files on the fly. Those files are not in the
cloud build, so their results were computed **once** on the full dataset by
`precompute.py` and baked into `_precomputed/*.json`. The values are identical
to what your local dashboard produces:

| Tab / endpoint | Used to read | Now reads | Verified value |
|---|---|---|---|
| Fraud rate by hour | `data/cleaned_transactions.csv` (672 MB) | `fraud_by_hour.json` (1.5 KB) | 5,078,336 rows, peak at hour 12 = 0.174% |
| Risk distribution | `results/final_risk_scores.csv` (416 MB) | `risk_distribution.json` (0.4 KB) | LOW 4,570,491 / MED 378,721 / HIGH 103,727 / CRIT 25,397 |
| Risk diagnostics | `results/final_risk_scores.csv` (416 MB) | `risk_diagnostics.json` (0.8 KB) | at 2.54% budget: ensemble 199 vs XGBoost 865 fraud caught |

Two more substitutions, both exact:

- **Live SHAP.** The `shap` package pulls in numba and llvmlite, roughly 180 MB
  of RAM at import — the difference between fitting in 512 MB and being killed.
  It was replaced with XGBoost's own `pred_contribs=True`, which runs the same
  TreeSHAP algorithm inside the library. Verified additive: the contributions
  plus the base value reproduce the model's raw margin to 6 decimal places, and
  the sigmoid of that reproduces the predicted probability exactly.
- **Random Forest details.** `random_forest.pkl` is 100.4 MB, over GitHub's
  per-file limit. Its hyper-parameters (200 trees, depth 12, min_samples_leaf 50,
  balanced class weights) and its top-15 feature importances were extracted from
  the pickle by `extract_rf.py` into `_precomputed/model_extras.json`, so that
  tab still shows everything. Only the ability to re-run predictions with that
  specific model is gone, and the dashboard never did that anyway.

The 5 GB of raw feature and training CSVs (`featured/`, `preprocessed/`,
`graph_features.csv`, …) were never read by the dashboard at all, so they simply
aren't here.

## Dependency pins — why these exact versions

| Pin | Reason |
|---|---|
| `numpy==2.1.3` | Your pickles reference `numpy._core.multiarray`, which only exists in numpy 2.x. numpy 1.24 fails every model load with `No module named 'numpy._core'`. |
| `scikit-learn==1.7.0` | The version your models were saved under. A different minor version loads with warnings and may behave differently. |
| no `lightgbm` | LightGBM needs `libgomp.so.1`, which is not on Render's Python image — it fails with `cannot open shared object file`. Its hyper-parameters and importances are read from `_precomputed/model_extras.json` instead, so that model panel is unchanged. |
| no `shap` | Pulls numba + llvmlite, ~180 MB of RAM. Replaced by XGBoost's built-in TreeSHAP. |

## Verified before shipping

Run against a real gunicorn server, the same command Render uses:

- 20/20 API endpoints returned HTTP 200; `/api/selftest` reported `all_ok: true`
- All 7 models return full detail panels, Random Forest included
- Live prediction reaches all four tiers — LOW (p=0.0095), MEDIUM (p=0.0143),
  HIGH (p=0.0412), CRITICAL (p=0.4805)
- Peak memory 286 MB against the 512 MB ceiling; boot 2.5 s
- Re-verified in a clean virtualenv built from `requirements.txt` alone:
  **20 models found, 0 load errors, 9 CSVs found, `xgboost_loaded: true`**
