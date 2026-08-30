# Rugby Bayesian Rankings Dashboard

An interactive web dashboard for exploring rugby team and player rankings from a hierarchical Bayesian model.

## Features

- **Team Rankings**: Offensive and defensive ratings with uncertainty quantification
- **Player Rankings**: Individual player effects across scoring types
- **Interactive Visualizations**: D3.js charts with hover tooltips and filtering
- **Match Explorer**: Browse historical match results
- **Offense vs Defense**: Scatter plot comparing team strengths
- **Responsive Design**: Works on desktop, tablet, and mobile

## Live Demo

🔗 [View Dashboard](https://transientlunatic.github.io/rugby-ranking/)

## Model

The rankings are generated from a hierarchical Bayesian model that captures:

- **Player Effects (β)**: Intrinsic ability that follows players across teams
- **Team Offensive Effects (γ)**: Team's ability to score
- **Team Defensive Effects (δ)**: Team's ability to prevent opponent scoring
- **Position Effects (θ)**: Base rates by jersey number
- **Home Advantage (η)**: Boost for playing at home

### Model Structure

```
log(λ) = α + β_player + γ_offense[team] - δ_defense[opponent]
         + θ_position + η_home × is_home + log(minutes/80)

N_score ~ Poisson(λ)
```

## Setup

### 1. Generate Data

First, fit the model and export data:

```bash
cd rugby-ranking
python export_dashboard_data.py
```

This creates JSON files in `dashboard/data/`:
- `team_offense.json` - Offensive rankings
- `team_defense.json` - Defensive rankings
- `player_rankings.json` - Player effects
- `match_stats.json` - Match results
- `team_stats.json` - Aggregated team statistics
- `summary.json` - Metadata

### 2. Test Locally

Serve the dashboard locally:

```bash
cd dashboard
python -m http.server 8000
```

Visit [http://localhost:8000](http://localhost:8000)

### 3. Deploy to GitHub Pages

#### Option A: Manual Deployment

1. Copy `dashboard/` contents to `docs/` in your repo root:
   ```bash
   mkdir -p docs
   cp -r dashboard/* docs/
   ```

2. Commit and push:
   ```bash
   git add docs/
   git commit -m "Deploy dashboard to GitHub Pages"
   git push origin main
   ```

3. Enable GitHub Pages:
   - Go to repo Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` / `docs` → Save

#### Option B: GitHub Actions (Automated)

Create `.github/workflows/deploy-dashboard.yml`:

```yaml
name: Deploy Dashboard

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly update

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: true
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Generate dashboard data
        run: |
          python export_dashboard_data.py

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dashboard
```

## File Structure

```
dashboard/
├── index.html           # Main dashboard page
├── css/
│   └── dashboard.css    # Custom styles
├── js/
│   └── dashboard.js     # D3.js visualizations
├── data/                # Generated data files
│   ├── team_offense.json
│   ├── team_defense.json
│   ├── player_rankings.json
│   ├── match_stats.json
│   ├── team_stats.json
│   └── summary.json
├── .nojekyll           # Disable Jekyll processing
└── README.md
```

## Technologies

- **Bootstrap 5**: Responsive UI framework
- **D3.js v7**: Data-driven visualizations
- **Bootstrap Icons**: Icon library
- **Vanilla JavaScript**: No framework overhead

## Customization

### Update Colors

Edit `dashboard/css/dashboard.css`:

```css
:root {
    --primary-color: #0d6efd;  /* Change primary color */
    --success-color: #198754;  /* Defense charts */
}
```

### Add New Visualizations

Edit `dashboard/js/dashboard.js`:

```javascript
function drawCustomChart(data) {
    // Your D3.js code here
}
```

### Modify Data Export

Edit `export_dashboard_data.py` to export additional statistics.

## Performance

- **Static Site**: No server required, fast loading
- **Data Files**: ~2-5 MB total (compressed JSON)
- **Caching**: Browser caches data files
- **CDN**: Bootstrap and D3.js loaded from CDN

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome)

## Updating Data

To update rankings with new match data:

```bash
# 1. Update data in Rugby-Data repo
cd Rugby-Data
git pull

# 2. Regenerate dashboard data
cd rugby-ranking
python export_dashboard_data.py

# 3. Deploy
git add dashboard/data/
git commit -m "Update rankings"
git push
```

## Troubleshooting

### CORS Errors

If loading locally without a server:
```bash
# Use a local server
python -m http.server 8000
```

### Missing Data

Ensure data files exist in `dashboard/data/`:
```bash
ls dashboard/data/
# Should see: team_offense.json, team_defense.json, etc.
```

### Slow Loading

- Check data file sizes
- Reduce `recent_seasons_only` in export script
- Limit number of teams/players exported

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes to dashboard files
4. Test locally
5. Submit a pull request

## License

See main repository LICENSE file.

## Links

- [Model Repository](https://github.com/transientlunatic/rugby-ranking)
- [Data Repository](https://github.com/transientlunatic/Rugby-Data)
- [Documentation](https://github.com/transientlunatic/rugby-ranking/blob/main/DEFENSIVE_MINIBATCH_UPDATE.md)
