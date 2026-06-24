# Census Analysis

Demographic analysis of a community survey (~1,300 respondents), producing 60+ charts across US-only and global scopes.

## What it covers

- Age, education, religious observance, and language proficiency distributions
- Cross-tabulations and trend lines across all demographic axes
- Occupation categorization from free-text responses (32 categories, multilingual)
- Birth-origin and generational shift analysis
- Deviation-from-baseline profiling by subgroup
- Student-adjusted employment breakdowns

## Usage

```bash
python3 analysis.py          # US-only (default)
python3 analysis.py global   # All respondents
```

Charts are saved to `charts/us/` or `charts/global/`.

## Requirements

- Python 3.8+
- pandas, matplotlib, numpy

## Data

The survey CSV is not included in this repository. Place your data file as `Raw_Census_2021.csv` in the project root to run the analysis.
