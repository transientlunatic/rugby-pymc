# Rugby Rankings Dashboard - Complete Guide

## Overview

This interactive web dashboard visualizes rugby team and player rankings from a hierarchical Bayesian model. Built with Bootstrap 5 and D3.js, it provides an intuitive interface for exploring offensive/defensive team strengths, player effects, and match statistics.

**Live Demo**: [https://transientlunatic.github.io/rugby-ranking/](https://transientlunatic.github.io/rugby-ranking/)

## Features

### 1. Dashboard Overview
- **Summary Cards**: Quick stats (seasons, matches, teams, players)
- **Model Information**: Description of Bayesian model architecture
- **Last Updated**: Timestamp of most recent data refresh

### 2. Team Rankings

#### Offensive Rankings
- Horizontal bar chart showing top N teams
- Error bars displaying 95% credible intervals
- Interactive tooltips with exact values
- Sortable table with uncertainty badges
- Filter by season, score type (tries/penalties/conversions), and rank limit

#### Defensive Rankings
- Similar to offensive rankings but for defensive ability
- Higher values = better defense (reduces opponent scoring more)
- Identify defensive specialists vs attacking teams

#### Offense vs Defense Scatter Plot
- Two-dimensional comparison of team strengths
- Four quadrants:
  - Top-right: Strong offense + strong defense (balanced powerhouses)
  - Top-left: Weak offense + strong defense (defensive specialists)
  - Bottom-right: Strong offense + weak defense (attacking teams)
  - Bottom-left: Weak offense + weak defense (struggling teams)
- Interactive hover for team identification

### 3. Player Rankings
- Bar chart of top 20 players by score type
- Searchable/filterable player list
- Separate rankings for tries, penalties, conversions
- Displays effect size with 95% credible intervals

### 4. Match Explorer
- Table of recent match results
- Filter by season and team
- Sortable by date
- Shows final scores and competition

## Quick Start

### Local Development

1. **Generate Data**:
   ```bash
   cd rugby-ranking
   python export_dashboard_data.py
   ```

2. **Serve Locally**:
   ```bash
   cd dashboard
   python -m http.server 8000
   ```

3. **Open Browser**:
   Navigate to [http://localhost:8000](http://localhost:8000)

### Deploy to GitHub Pages

#### Method 1: Manual

```bash
# 1. Ensure dashboard/data/ contains JSON files
ls dashboard/data/

# 2. Copy to docs/ folder (if using docs/ for Pages)
mkdir -p docs
cp -r dashboard/* docs/

# 3. Commit and push
git add docs/ dashboard/
git commit -m "Update dashboard"
git push origin main

# 4. Enable GitHub Pages
# Settings → Pages → Source: main branch / docs folder
```

#### Method 2: Automated (GitHub Actions)

The included workflow (`.github/workflows/deploy-dashboard.yml`) automatically:
- Triggers on push to main, weekly schedule, or manual dispatch
- Generates fresh data from model
- Deploys to `gh-pages` branch
- Updates live site at `https://USERNAME.github.io/rugby-ranking/`

**Enable**: Push the workflow file, then check Actions tab in GitHub.

## Architecture

### Data Flow

```
Raw Match Data → PyMC Model → VI Inference → Export Script → JSON Files → Dashboard
```

1. **Match Data**: Loaded from Rugby-Data repository
2. **Model Fitting**: Hierarchical Bayesian model with VI (ADVI)
3. **Export**: [export_dashboard_data.py](export_dashboard_data.py) extracts rankings
4. **JSON Files**: Static data files in `dashboard/data/`
5. **Dashboard**: Client-side JavaScript loads and visualizes data

### File Structure

```
rugby-ranking/
├── dashboard/
│   ├── index.html              # Main HTML page
│   ├── css/
│   │   └── dashboard.css       # Custom styles
│   ├── js/
│   │   └── dashboard.js        # D3.js visualizations & logic
│   ├── data/                   # Generated JSON files
│   │   ├── team_offense.json   # Team offensive rankings
│   │   ├── team_defense.json   # Team defensive rankings
│   │   ├── player_rankings.json # Player effects
│   │   ├── match_stats.json    # Match results
│   │   ├── team_stats.json     # Aggregated team stats
│   │   └── summary.json        # Metadata
│   ├── .nojekyll              # Disable Jekyll
│   └── README.md
├── export_dashboard_data.py    # Data export script
├── .github/workflows/
│   └── deploy-dashboard.yml    # Auto-deployment
└── DASHBOARD.md                # This file
```

### Data Schema

#### team_offense.json / team_defense.json
```json
[
  {
    "team": "Leinster Rugby",
    "season": "2023-2024",
    "score_type": "tries",
    "offense_mean": 0.298,
    "offense_std": 0.082,
    "offense_lower": 0.137,
    "offense_upper": 0.459
  },
  ...
]
```

#### player_rankings.json
```json
[
  {
    "player": "Antoine Dupont",
    "score_type": "tries",
    "effect_mean": 0.542,
    "effect_std": 0.091,
    "effect_lower": 0.364,
    "effect_upper": 0.720
  },
  ...
]
```

#### match_stats.json
```json
[
  {
    "match_id": "premiership_2023-2024_45",
    "season": "2023-2024",
    "team": "Saracens",
    "opponent": "Leicester Tigers",
    "team_score": 24,
    "opponent_score": 17,
    "team_tries": 3,
    "team_penalties": 3,
    "team_conversions": 2,
    "date": "2024-03-15T15:00:00",
    "competition": "premiership"
  },
  ...
]
```

## Customization

### Branding

**Logo**: Replace navbar brand in `index.html`:
```html
<a class="navbar-brand" href="#">
    <img src="logo.png" height="30"> Your Rugby League
</a>
```

**Colors**: Edit CSS variables in `dashboard.css`:
```css
:root {
    --primary-color: #0d6efd;  /* Main brand color */
    --success-color: #198754;  /* Defense charts */
    --danger-color: #dc3545;   /* Negative indicators */
}
```

### Add New Score Types

1. **Export Script**: Modify `export_dashboard_data.py`:
   ```python
   for score_type in ["tries", "penalties", "conversions", "drop_goals"]:
       # Export logic
   ```

2. **HTML**: Add option to dropdowns:
   ```html
   <option value="drop_goals">Drop Goals</option>
   ```

3. **JavaScript**: No changes needed (dynamically loads score types)

### Add New Visualizations

Create custom D3.js chart in `dashboard.js`:

```javascript
function drawCustomChart(data) {
    const svg = d3.select('#custom-chart')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Your D3.js code here
}
```

Add HTML container:
```html
<div id="custom-chart"></div>
```

## Performance Optimization

### Data Size Reduction

**Limit Seasons** in `export_dashboard_data.py`:
```python
export_dashboard_data(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    recent_seasons_only=2  # Only last 2 seasons
)
```

**Limit Rankings**:
```python
rankings = model.get_team_rankings(
    trace=trace,
    season=season,
    score_type=score_type,
    top_n=30  # Reduce from 50
)
```

### Caching

GitHub Pages automatically caches static files. For manual hosting:

**Nginx** (add to config):
```nginx
location ~* \.(json)$ {
    expires 1d;
    add_header Cache-Control "public, immutable";
}
```

**Apache** (.htaccess):
```apache
<filesMatch "\.(json)$">
    Header set Cache-Control "max-age=86400, public"
</filesMatch>
```

### Lazy Loading

Load data only when tabs are activated:

```javascript
document.getElementById('defense-tab').addEventListener('shown.bs.tab', () => {
    if (!state.defenseLoaded) {
        updateDefenseVisualizations();
        state.defenseLoaded = true;
    }
});
```

## Browser Compatibility

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 90+ | ✅ Full |
| Firefox | 88+ | ✅ Full |
| Safari | 14+ | ✅ Full |
| Edge | 90+ | ✅ Full |
| Mobile Safari | iOS 14+ | ✅ Full |
| Chrome Mobile | Android 90+ | ✅ Full |

**Polyfills Not Needed**: Modern ES6+ features only (Promise, fetch, arrow functions).

## Troubleshooting

### Issue: CORS Errors (Local Development)

**Symptom**: Console error `Access to fetch blocked by CORS policy`

**Solution**: Use a local server instead of `file://`
```bash
python -m http.server 8000
# or
npx serve dashboard
```

### Issue: Data Not Loading

**Symptom**: Dashboard shows "Loading..." indefinitely

**Check**:
1. Data files exist: `ls dashboard/data/*.json`
2. JSON is valid: `python -m json.tool dashboard/data/summary.json`
3. Browser console for errors: F12 → Console

**Solution**: Regenerate data
```bash
python export_dashboard_data.py
```

### Issue: Charts Not Rendering

**Symptom**: Empty white boxes where charts should be

**Check**:
1. Browser console for D3.js errors
2. Container widths: `console.log(container.clientWidth)`
3. Data format matches expected schema

**Solution**: Check D3.js loading
```html
<script src="https://d3js.org/d3.v7.min.js"></script>
```

### Issue: GitHub Pages 404

**Symptom**: Site not accessible at username.github.io/rugby-ranking

**Solutions**:
1. Enable Pages in Settings → Pages
2. Check branch is `gh-pages` or `main/docs`
3. Ensure `.nojekyll` file exists
4. Wait 5-10 minutes for deployment

## Advanced Usage

### Custom Data Sources

Modify `export_dashboard_data.py` to load from different sources:

```python
# Load from database
import psycopg2
conn = psycopg2.connect("dbname=rugby user=postgres")
df = pd.read_sql("SELECT * FROM matches", conn)

# Load from API
import requests
response = requests.get("https://api.rugby.com/matches")
df = pd.DataFrame(response.json())
```

### Multiple Dashboards

Create separate dashboards per competition:

```bash
python export_dashboard_data.py --competition premiership --output dashboard/premiership
python export_dashboard_data.py --competition top14 --output dashboard/top14
```

Update HTML to load from subdirectories:
```javascript
const competition = 'premiership';
const data = await d3.json(`data/${competition}/team_offense.json`);
```

### API Backend (Optional)

For dynamic updates without regenerating static files, create Flask API:

```python
from flask import Flask, jsonify
from rugby_ranking.model.core import RugbyModel

app = Flask(__name__)

@app.route('/api/teams/<season>/<score_type>')
def get_teams(season, score_type):
    # Load model and return rankings
    rankings = model.get_team_rankings(season=season, score_type=score_type)
    return jsonify(rankings.to_dict('records'))

app.run(port=5000)
```

Update dashboard JavaScript:
```javascript
const data = await fetch(`/api/teams/${season}/${scoreType}`).then(r => r.json());
```

## Contributing

### Adding Features

1. Fork repository
2. Create feature branch: `git checkout -b feature/new-chart`
3. Make changes to dashboard files
4. Test locally: `python -m http.server`
5. Commit: `git commit -m "Add new chart type"`
6. Push and create PR

### Reporting Issues

Open issue with:
- Browser and version
- Steps to reproduce
- Expected vs actual behavior
- Console errors (F12)
- Screenshot (if visual issue)

## FAQ

**Q: Can I use this for my own rugby data?**
A: Yes! Modify `export_dashboard_data.py` to load your data, then regenerate JSON files.

**Q: How often should I update?**
A: Weekly is recommended (automated via GitHub Actions schedule).

**Q: Can I add more score types?**
A: Yes, edit export script and HTML dropdowns. Dashboard auto-adapts.

**Q: Is there a way to compare players directly?**
A: Not currently, but you can add a comparison view by forking and customizing.

**Q: Can I embed this in another site?**
A: Yes, use iframe: `<iframe src="https://username.github.io/rugby-ranking/" width="100%" height="800"></iframe>`

## Resources

- **Bootstrap Documentation**: [getbootstrap.com](https://getbootstrap.com/docs/5.3/)
- **D3.js Examples**: [d3-graph-gallery.com](https://d3-graph-gallery.com/)
- **GitHub Pages Guide**: [docs.github.com/pages](https://docs.github.com/en/pages)
- **PyMC Documentation**: [pymc.io](https://www.pymc.io/)

## License

See main repository LICENSE file.

## Support

- **Issues**: [GitHub Issues](https://github.com/transientlunatic/rugby-ranking/issues)
- **Discussions**: [GitHub Discussions](https://github.com/transientlunatic/rugby-ranking/discussions)
- **Model Docs**: [DEFENSIVE_MINIBATCH_UPDATE.md](DEFENSIVE_MINIBATCH_UPDATE.md)
