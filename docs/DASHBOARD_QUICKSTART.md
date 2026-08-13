# Dashboard Quick Start Guide

Get your rugby rankings dashboard live in 5 minutes!

## Step 1: Generate Data (2 minutes)

```bash
cd rugby-ranking
python export_dashboard_data.py
```

This creates `dashboard/data/*.json` files with model predictions.

## Step 2: Test Locally (30 seconds)

```bash
cd dashboard
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Step 3: Deploy to GitHub Pages (2 minutes)

### Option A: Manual

```bash
# Enable GitHub Pages
# Go to: Settings → Pages → Source: main branch / dashboard folder
# OR copy to docs/:

mkdir -p docs
cp -r dashboard/* docs/
git add docs/
git commit -m "Deploy dashboard"
git push origin main
```

### Option B: Automated

```bash
# Push the GitHub Actions workflow
git add .github/workflows/deploy-dashboard.yml
git commit -m "Add automated deployment"
git push origin main

# Dashboard auto-deploys on every push!
```

## Step 4: View Live (instant)

Your dashboard is live at:

```
https://YOUR_USERNAME.github.io/rugby-ranking/
```

## File Checklist

Ensure these files exist:

```
✅ dashboard/index.html
✅ dashboard/css/dashboard.css
✅ dashboard/js/dashboard.js
✅ dashboard/data/summary.json
✅ dashboard/data/team_offense.json
✅ dashboard/data/team_defense.json
✅ dashboard/data/player_rankings.json
✅ dashboard/data/match_stats.json
✅ dashboard/data/team_stats.json
✅ dashboard/.nojekyll
```

## Troubleshooting

### "Data not loading"

```bash
# Regenerate data
python export_dashboard_data.py

# Check files created
ls -lh dashboard/data/
```

### "CORS errors locally"

```bash
# Use a server, not file://
python -m http.server 8000
```

### "404 on GitHub Pages"

1. Check Settings → Pages is enabled
2. Wait 5-10 minutes for first deploy
3. Ensure `.nojekyll` file exists

## Next Steps

- **Customize**: Edit `dashboard/css/dashboard.css` for colors
- **Add Data**: Modify `export_dashboard_data.py` for more seasons
- **Automate**: Set up GitHub Actions for weekly updates

## Full Documentation

See [DASHBOARD.md](DASHBOARD.md) for complete guide.
