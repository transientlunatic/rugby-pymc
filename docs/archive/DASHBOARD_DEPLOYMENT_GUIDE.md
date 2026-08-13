# Complete Dashboard Deployment Guide

## Quick Start (5 Minutes)

### Prerequisites
- ✅ Model checkpoint trained and saved (or will be trained on first run)
- ✅ Rugby-Data directory with JSON match files
- ✅ Python environment with all dependencies installed

### One-Command Deployment

```bash
# From rugby-ranking directory
python tests/export_dashboard_data.py && python -m http.server 8000 --directory dashboard
```

Then open: **http://localhost:8000**

---

## Detailed Walkthrough

### 1. Train or Load Model

**Option A: Use Existing Checkpoint**
```bash
cd rugby-ranking

# export_dashboard_data.py will load the checkpoint automatically:
# checkpoint_name="time_model_v1"  (or modify in script)
```

**Option B: Train New Model**
```bash
cd rugby-ranking

python train_model.py \
  --last-seasons 5 \
  --data-dir ../Rugby-Data \
  --method vi \
  --save-as dashboard_export \
  --model time-varying

# This creates: models/checkpoints/dashboard_export/
```

### 2. Export Dashboard Data

```bash
cd rugby-ranking

python tests/export_dashboard_data.py
```

**Expected Output:**
```
======================================================================
EXPORTING DASHBOARD DATA
======================================================================

Loading data...
Loaded 50,000 observations

Filtering to 3 recent seasons: ['2022-2023', '2023-2024', '2024-2025']
Using 48,000 observations

Loading checkpoint: time_model_v1

Exporting rankings and statistics...
  - Team offensive rankings...
  - Team defensive rankings...
  - Player rankings...
  - Match statistics...
  - Summary statistics...
  - Team aggregated statistics...
  - Team strength series...
  - Team finish positions...
  - Upcoming predictions...
  - Paths to victory...
  - Squad depth...

======================================================================
EXPORT COMPLETE
======================================================================

Files written to: dashboard/data/
  - team_offense.json
  - team_defense.json (if defense enabled)
  - player_rankings.json
  - match_stats.json
  - team_stats.json
  - summary.json
  - team_strength_series.json (NEW)
  - team_finish_positions.json (NEW)
  - upcoming_predictions.json (NEW)
  - paths_to_victory.json (NEW)
  - squad_depth.json (NEW)

Ready for dashboard deployment!
```

**Time to Complete**: 5-10 minutes (depending on dataset size)

### 3. Serve Dashboard

```bash
# Option 1: Simple HTTP Server (for local development)
cd rugby-ranking/dashboard
python -m http.server 8000
# Visit: http://localhost:8000

# Option 2: With VS Code Simple Browser
# Already opened in editor

# Option 3: Live Server (if extension installed)
# Right-click index.html → Open with Live Server
```

### 4. Verify All Sections Load

Open dashboard at **http://localhost:8000** and check each section:

| Section | Expected Behavior |
|---------|-------------------|
| **Offensive Rankings** | Shows team names with try/penalty/conversion bars + CI |
| **Defensive Rankings** | Shows defensive strength by team |
| **Player Rankings** | Shows top try-scorers |
| **Team Comparison** | Shows offensive vs defensive scatter plot |
| **Trends** | Dropdown for team select; shows strength over seasons |
| **Positions** | Dropdown for competition; shows position rankings |
| **Predictions** | Table with upcoming matches and win probabilities |
| **Paths** | Narrative text + table of critical games with MI scores |
| **Squads** | Dropdown for team select; shows depth chart table |

---

## Dashboard Sections in Detail

### 1. Offensive Rankings (Card #1)
- **What it shows**: Top 20 teams ranked by offensive strength (tries per match)
- **Data source**: `team_offense.json`
- **Interaction**: Score type filter (tries/penalties/conversions)
- **Metric**: Effect size from Bayesian model with 95% credible intervals

### 2. Defensive Rankings (Card #2)
- **What it shows**: Top 20 teams ranked by defensive strength (tries conceded)
- **Data source**: `team_defense.json`
- **Interaction**: Score type filter
- **Metric**: Effect size (higher = less likely to concede)

### 3. Player Rankings (Card #3)
- **What it shows**: Top 30 try-scorers across all seasons
- **Data source**: `player_rankings.json`
- **Interaction**: Score type filter
- **Metric**: Player-level try-scoring rate effect

### 4. Team Comparison (Card #4)
- **What it shows**: Offensive vs Defensive strength (scatter plot)
- **Data source**: `team_offense.json` + `team_defense.json`
- **Interaction**: Season filter
- **Insight**: Teams in top-right are strongest (strong offense + defense)

### 5. **Trends** (Card #5) - NEW
- **What it shows**: Team strength over seasons (time-series)
- **Data source**: `team_strength_series.json`
- **Interaction**: Team select dropdown, score type filter
- **Chart Type**: Line chart with optional reversal (for rankings)
- **Insight**: Identify improving/declining teams across seasons

### 6. **Finish Positions** (Card #6) - NEW
- **What it shows**: Historical final league position over seasons
- **Data source**: `team_finish_positions.json`
- **Interaction**: Team select dropdown, competition filter
- **Chart Type**: Line chart with y-reversed (lower position = better)
- **Insight**: Track team consistency in final standings

### 7. **Predictions** (Card #7) - NEW
- **What it shows**: Match predictions for upcoming fixtures
- **Data source**: `upcoming_predictions.json`
- **Interaction**: Season filter, competition filter
- **Columns**: Date, Home, Prediction, Away, Win %, Competition
- **Insight**: Model's predicted outcomes and confidence

### 8. **Paths to Victory** (Card #8) - NEW
- **What it shows**: Critical games and narrative for achieving target positions
- **Data source**: `paths_to_victory.json`
- **Interaction**: Team select dropdown, competition select
- **Output Format**:
  - **Narrative** (pre-formatted text): "To finish top-2, Scotland needs..."
  - **Critical Games Table**: Games ranked by mutual information score
- **Insight**: Which games matter most for achieving team goals

### 9. **Squad Depth** (Card #9) - NEW
- **What it shows**: Squad composition and position strength
- **Data source**: `squad_depth.json`
- **Interaction**: Team select dropdown, season filter
- **Output Format**:
  - **Summary**: Overall strength score and depth score
  - **Depth Chart**: Position × Top 3 players with ratings
- **Insight**: Identify weak positions or injury-prone areas

---

## Customization Options

### Modify Export Parameters
Edit `export_dashboard_data.py` line 550+:

```python
if __name__ == "__main__":
    DATA_DIR = Path("../Rugby-Data")
    OUTPUT_DIR = Path("dashboard/data")

    export_dashboard_data(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_name="time_model_v1",      # Change checkpoint name
        recent_seasons_only=5                 # Export more seasons
    )
```

### Modify Dashboard Appearance
Edit `dashboard/css/style.css`:
- Colors, fonts, layout
- Card sizes and spacing
- Chart dimensions

### Add New Visualizations
Edit `dashboard/js/dashboard.js`:
- Add new `populate*()` function for dropdowns
- Add new `update*()` function for chart rendering
- Wire to event listeners

---

## Troubleshooting

### Problem: "models/checkpoints/time_model_v1/ not found"
**Solution**: Specify correct checkpoint name or omit it to train new model
```python
# In export_dashboard_data.py line 550:
checkpoint_name=None  # Will train new model
# or
checkpoint_name="your_checkpoint_name"
```

### Problem: "dashboard/data/ directory not found"
**Solution**: Script creates it automatically, ensure write permissions
```bash
mkdir -p dashboard/data
chmod 755 dashboard/data
```

### Problem: Squad sections show empty
**Solution**: Ensure squad CSV files exist in `squads/` directory
```bash
ls squads/*.csv  # Check if files exist

# If not, add squads:
rugby-ranking squad input --team "Scotland" --season "2024-2025"
```

### Problem: Paths to victory section errors
**Solution**: Requires remaining fixtures; may show empty for completed seasons
- Works best during active tournament
- Falls back to heuristic if samples unavailable

### Problem: Dashboard loads but charts don't render
**Solution**: Check browser console for errors
```javascript
// Open DevTools (F12) → Console tab
// Look for JSON loading errors
// Verify loadJsonSafe() is working
```

### Problem: Slow export (>20 minutes)
**Solution**: Reduce dataset size
```python
# In export_dashboard_data.py:
recent_seasons_only=2  # Export fewer seasons
```

---

## Production Deployment

### Deploy to Web Server (Ubuntu/Nginx)

```bash
# 1. Build dashboard on server
cd /opt/rugby-ranking
python tests/export_dashboard_data.py

# 2. Configure Nginx to serve static files
sudo nano /etc/nginx/sites-available/rugby-dashboard

# Add:
server {
    listen 80;
    server_name your-domain.com;
    root /opt/rugby-ranking/dashboard;
    index index.html;
    
    # Cache JSON for 1 hour
    location /data/ {
        expires 1h;
    }
}

# 3. Enable and restart
sudo ln -s /etc/nginx/sites-available/rugby-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Automated Daily Export (Cron)

```bash
# Edit crontab
crontab -e

# Add line (runs daily at 2 AM):
0 2 * * * cd /opt/rugby-ranking && python tests/export_dashboard_data.py >> /var/log/rugby-export.log 2>&1

# Verify:
tail -f /var/log/rugby-export.log
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python tests/export_dashboard_data.py && python -m http.server 8000 --directory dashboard"]
```

---

## Performance Notes

### Export Speed by Dataset Size
| Seasons | Observations | Export Time |
|---------|--------------|-------------|
| 1       | 10,000       | 2-3 min     |
| 3       | 30,000       | 5-8 min     |
| 5       | 50,000       | 10-15 min   |
| 10+     | 100,000+     | 20+ min     |

### Dashboard Performance
| Metric | Value |
|--------|-------|
| Page Load | <1s |
| Chart Render | <100ms each |
| Browser Memory | 50-100 MB |
| Network Transfer | ~2-5 MB |

### Optimization Tips
1. **Server-side**: Cache checkpoint in memory, precompute rankings
2. **Client-side**: Use browser DevTools to profile chart rendering
3. **Network**: Gzip JSON files in nginx
4. **Database**: Store precomputed exports in DB instead of JSON

---

## Integration Examples

### Embed in Blog Post
```html
<!-- Copy rugby_charts.js and include -->
<script src="rugby-ranking/dashboard/js/rugby_charts.js"></script>

<div id="my-chart"></div>

<script>
// Load data from dashboard
fetch('dashboard/data/team_strength_series.json')
  .then(r => r.json())
  .then(data => {
    RugbyCharts.renderLineChart({
      container: '#my-chart',
      data: data.filter(d => d.team === 'Scotland'),
      xKey: 'season',
      yKey: 'offense_mean'
    });
  });
</script>
```

### Export to CSV
```python
import pandas as pd

# Load dashboard data
strength = pd.read_json('dashboard/data/team_strength_series.json')
predictions = pd.read_json('dashboard/data/upcoming_predictions.json')

# Export to CSV
strength.to_csv('team_strength.csv', index=False)
predictions.to_csv('upcoming_matches.csv', index=False)
```

---

## Summary

**Complete Dashboard Ready to Deploy:**
1. ✅ Run `python tests/export_dashboard_data.py` to export data
2. ✅ Run `python -m http.server 8000 --directory dashboard` to serve
3. ✅ Open http://localhost:8000 to view all 11 sections
4. ✅ Interact with filters and dropdowns to explore data
5. ✅ Export visualizations or data as needed

**All systems go!** 🚀
