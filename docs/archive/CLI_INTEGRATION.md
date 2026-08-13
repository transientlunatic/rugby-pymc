# Phase 5 CLI Integration Complete

**Date:** 2026-02-13

## Summary

Phase 5 Squad Analysis functionality is now fully integrated with the rugby-ranking CLI. Users can access all Phase 5 features through command-line commands.

## Available Commands

### 1. Squad Input
```bash
rugby-ranking squad input --team "Scotland" --season "2025-2026"
rugby-ranking squad input --team "Scotland" --season "2025-2026" --file squad.txt --format wikipedia
```

**What it does:** Parse and save squad data from Wikipedia format, CSV, or text

---

### 2. Squad Analysis
```bash
rugby-ranking squad analyze --team "Scotland" --season "2025-2026" --checkpoint latest
rugby-ranking squad analyze --team "Scotland" --season "2025-2026" --detailed
```

**What it does:** Analyze squad strength, depth, and position-by-position breakdown

**Output:**
- Overall squad strength and depth scores
- Position-by-position strength analysis
- Vulnerable positions identification
- Most likely starting XV and bench
- Saves detailed report to `reports/` directory

---

### 3. Squad Comparison
```bash
rugby-ranking squad compare --tournament six-nations --season "2025-2026" --checkpoint latest
```

**What it does:** Compare all squads in a tournament

**Output:**
- Overall squad rankings by strength and depth
- Team-by-team analysis
- Saves comparison report to `reports/` directory

---

### 4. Lineup Prediction (NEW)
```bash
rugby-ranking squad lineup --team "Scotland" --season "2025-2026" --checkpoint latest
rugby-ranking squad lineup --team "Scotland" --season "2025-2026" --unavailable "Finn Russell" "Stuart Hogg"
```

**What it does:** Predict most likely starting XV and bench

**Features:**
- Optimization-based selection
- Positional coverage validation
- Support for injury scenarios (--unavailable flag)
- Shows starting XV by position number
- Shows 8-player bench

**Example Output:**
```
PREDICTED LINEUP: Scotland
======================================================================

STARTING XV
----------------------------------------------------------------------
 1. Prop             Pierre Schoeman
 2. Hooker           George Turner
 3. Prop             Zander Fagerson
...

BENCH
----------------------------------------------------------------------
16. Dave Cherry
17. Rory Sutherland
...

Total Rating: 0.125
Coverage Valid: ✓
```

---

### 5. Critical Players Analysis (NEW)
```bash
rugby-ranking squad critical-players --team "Scotland" --season "2025-2026" --top 10
```

**What it does:** Identify most critical/irreplaceable players

**Features:**
- Analyzes impact of losing each player
- Identifies most likely replacement
- Calculates criticality score (0-100%)
- Shows top N most critical players

**Example Output:**
```
MOST CRITICAL PLAYERS: Scotland
======================================================================

Rank   Player                    Position     Criticality  Replacement
----------------------------------------------------------------------
1      Finn Russell              Fly-half     89%          Adam Hastings
2      Stuart Hogg               Fullback     82%          Blair Kinghorn
3      Hamish Watson             Flanker      76%          Luke Crosbie
...
```

**Criticality Scale:**
- 100% = Irreplaceable (no valid replacement)
- 80-99% = Extremely critical
- 60-79% = Very important
- 40-59% = Moderate impact
- 0-39% = Easily replaceable (good depth)

---

### 6. Squad Robustness Analysis (NEW)
```bash
rugby-ranking squad robustness --team "Scotland" --season "2025-2026"
rugby-ranking squad robustness --team "Scotland" --season "2025-2026" --simulations 200 --injury-prob 0.20
```

**What it does:** Simulate random injuries to assess squad resilience

**Features:**
- Monte Carlo injury simulations
- Measures impact on team strength
- Identifies positions most vulnerable to injuries
- Calculates overall robustness score

**Parameters:**
- `--simulations`: Number of scenarios (default: 100)
- `--injury-prob`: Probability each player is injured (default: 0.15 = 15%)

**Example Output:**
```
SQUAD ROBUSTNESS ANALYSIS: Scotland
======================================================================

Robustness Score: 78%
  (Higher is better: team maintains strength despite injuries)

Average Impact:   0.042 rating points
Std Deviation:    0.028
Best Case:        0.000 (no/minor injuries)
Worst Case:       0.156 (severe injuries)

POSITIONS MOST VULNERABLE TO INJURIES
----------------------------------------------------------------------
  Fly-half             Average Impact: 0.089
  Loosehead Prop       Average Impact: 0.067
  Scrum-half           Average Impact: 0.054
```

**Robustness Score Interpretation:**
- 90-100%: Excellent depth across squad
- 70-89%: Good depth, can handle typical injuries
- 50-69%: Moderate depth, vulnerable to injuries
- <50%: Poor depth, significant risk

---

## Usage Examples

### Pre-Tournament Analysis
```bash
# 1. Input all squads (once squads are announced)
rugby-ranking squad input --team "Scotland" --season "2025-2026"
rugby-ranking squad input --team "Ireland" --season "2025-2026"
# ... repeat for all teams

# 2. Compare all squads
rugby-ranking squad compare --tournament six-nations --season "2025-2026"

# 3. Detailed analysis of your team
rugby-ranking squad analyze --team "Scotland" --season "2025-2026" --detailed
rugby-ranking squad critical-players --team "Scotland" --season "2025-2026"
rugby-ranking squad robustness --team "Scotland" --season "2025-2026"
```

### Weekly Match Preparation
```bash
# 1. Predict lineup (if injuries/changes)
rugby-ranking squad lineup --team "Scotland" --season "2025-2026" --unavailable "Finn Russell"

# 2. Check impact of key injuries
rugby-ranking squad critical-players --team "Scotland" --season "2025-2026" --top 5

# 3. Update robustness if squad changes
rugby-ranking squad robustness --team "Scotland" --season "2025-2026"
```

### Blog Post Generation
```bash
# 1. Generate squad comparison (tournament preview)
rugby-ranking squad compare --tournament six-nations --season "2025-2026"
# Output saved to: reports/six-nations_2025-2026_comparison.txt

# 2. Generate team analysis (individual team preview)
rugby-ranking squad analyze --team "Scotland" --season "2025-2026" --detailed
# Output saved to: reports/scotland_2025-2026_analysis.txt

# Can copy these reports directly into blog posts!
```

---

## File Structure

### Squad Files
Squads are stored in CSV format in the `squads/` directory:
```
squads/
  scotland_2025-2026.csv
  ireland_2025-2026.csv
  england_2025-2026.csv
  ...
```

Alternatively, squads can be in JSON format:
```
squads/
  2026_six_nations_championship_squads.json  # All teams in one file
```

### Output Reports
Analysis reports are saved in the `reports/` directory:
```
reports/
  scotland_2025-2026_analysis.txt           # Individual team analysis
  six-nations_2025-2026_comparison.txt      # Tournament comparison
```

---

## Model Checkpoints

Most commands require a trained model checkpoint. Common checkpoints:

- `latest`: Most recent model (default for most commands)
- `international-mini5`: International matches only (default for squad analysis)
- `joint_model_v2`: Full joint model (all score types)

Specify with `--checkpoint <name>` flag.

---

## Integration with Rugby-Data

The CLI automatically searches for squad files in multiple locations:
1. `squads/` (local directory)
2. Rugby-Data config directory (if rugby package is installed)
3. `../Rugby-Data/squads` (sibling repository)

This allows seamless integration with the Rugby-Data repository.

---

## Error Handling

### Squad Not Found
```
✗ Squad file not found for Scotland (2025-2026)

First input the squad using:
  rugby-ranking squad input --team "Scotland" --season 2025-2026
```

**Solution:** Input the squad first using `squad input` command

### Insufficient Players
```
✗ Error predicting lineup: Insufficient players: 18 available, need at least 23
```

**Solution:** Ensure squad has at least 23 players (15 starters + 8 bench)

### Player Not Found
```
Warning: Could not match 3/35 players to model
  Unmatched players: John Smith, Jane Doe, ...
```

**Explanation:** These players haven't played enough matches to be in the model. They'll be assigned league-average ratings.

---

## Future Enhancements

Potential additions for Phase 6+:

1. **Squad-based match predictions:**
   ```bash
   rugby-ranking predict --home "Scotland" --away "Ireland" --use-squads
   ```

2. **Export to markdown/HTML:**
   ```bash
   rugby-ranking squad analyze --team "Scotland" --export markdown -o blog/scotland.md
   ```

3. **Injury scenario comparison:**
   ```bash
   rugby-ranking squad scenario --team "Scotland" --compare "Russell,Hogg" "Watson,Turner"
   ```

4. **Interactive lineup builder:**
   ```bash
   rugby-ranking squad lineup --team "Scotland" --interactive
   ```

---

## Summary

Phase 5 CLI integration provides **6 commands** covering:
- ✅ Squad input and parsing
- ✅ Strength and depth analysis
- ✅ Tournament-wide comparisons
- ✅ Lineup prediction (with injury scenarios)
- ✅ Critical player identification
- ✅ Squad robustness assessment

All functionality is accessible via intuitive command-line interface, with outputs suitable for direct use in blog posts and reports.

**Status: ✅ Complete and Ready for Use**
