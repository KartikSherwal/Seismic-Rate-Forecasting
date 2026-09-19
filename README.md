# Earthquake Seismicity Forecasting

Forecasting regional monthly earthquake activity using the USGS global catalog built after
diagnosing and fixing a data leakage problem in an earlier magnitude prediction attempt.

The notebook includes both parts of that story: the leakage diagnosis (why predicting a single
earthquake's magnitude from network metadata doesn't work) and the corrected approach (forecasting
next-month event counts and the probability of a significant event, per region, benchmarked
against several baselines before introducing an LSTM).

---

## Why this framing

An earlier version of this project tried to predict an earthquake's **magnitude** from its
location and its seismic network metadata (station count, azimuthal gap, distance to nearest
station, location fit residual). That framing has a leakage problem: those metadata features are
only known *after* a quake is detected and processed a bigger quake is picked up by more
stations, so the model was partly reading the answer off the back of the card. It also relied on a
random train/test split on time-ordered data, which further inflates the score.

Section 3 of the notebook measures this directly  comparing a random split against a strict
temporal split, running permutation importance, and testing a geo-only ablation before moving to
a corrected, deployable framing: **forecast rates, not single events**. The rest of the notebook
builds that forecasting pipeline on the full USGS catalog, binning the globe into 5 degree x 5
degree monthly cells and predicting next month event counts and the probability of a significant
(M >= 5.5) event.

---

## Repo structure

```
.
├── data/                              # Raw/cached USGS catalog data
├── outputs/                           # Saved model artifacts, panel data, and figures
│   ├── preprocessing_and_baselines.joblib
│   ├── monthly_panel.csv
│   └── figures/
├── earthquake_forecasting_lstm.ipynb  # Full pipeline: USGS data pull, EDA, leakage diagnosis,
│                                       #   panel construction, feature engineering, baselines, LSTM
├── predict_lstm.py                    # Interactive CLI: forecast next-month seismicity for a
│                                       #   region, from either the saved panel or your own input
├── requirements.txt
├── .gitignore
└── LICENSE
```

`data/` and `outputs/` are included here so `predict_lstm.py` works right after cloning, without
needing to re-run the full notebook first. Re-running the notebook will regenerate them.

---

## Quickstart

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### 1. Run the notebook

```bash
jupyter lab earthquake_forecasting_lstm.ipynb
```

On first run it downloads the full USGS catalog (5-15 minutes, cached afterward under `data/`).
Everything runs on scikit-learn alone except the LSTM cells, which need `pip install tensorflow`.

### 2. Forecast next-month regional activity

```bash
python predict_lstm.py
```

This walks you through either looking up a region already in the saved panel, or typing in your
own recent monthly earthquake counts (e.g. numbers pulled from the
[USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/) for a region you care
about). For scripting, `python predict_lstm.py --cell "35_140"` looks up a saved region directly,
and `python predict_lstm.py --list-cells` shows which regions are available.

---

## Methodology summary

| Stage | What was done |
|---|---|
| Data | USGS FDSN Event API, M >= 4.5, 1990 to present |
| Leakage diagnosis | Random vs. temporal split comparison, permutation importance, geo-only ablation |
| EDA | Catalog completeness, Gutenberg-Richter fit (b-value), magnitude of completeness, depth structure, spatial distribution |
| Reframing | 5x5 degree spatial cells x monthly bins, forming a genuine time-series forecasting panel |
| Features | 17 engineered features per timestep: rolling counts/energy (3/6/12-month), volatility, rolling b-value, months since last significant event, seasonal encodings |
| Split | Strictly chronological (train up to 2013, validation 2014-2018, test 2019 onward); scaler fit on train only |
| Baselines | Persistence, rolling mean, per-cell climatology, gradient boosting |
| Model | Stacked LSTM (64 -> 32) with dropout, Huber loss (regression) / class-weighted BCE (classification) |
| Evaluation | MAE/RMSE/R-squared for counts; ROC-AUC, PR-AUC, Brier score, and Brier skill score vs. climatology for classification |

Full reasoning for each choice including why persistence and climatology are strong baselines
here, and what a good vs. weak result actually looks like is in the markdown cells of the
notebook.

---

## Results

Test window (2019 onward), next-month event count forecasting:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence | 2.885 | 12.788 | -0.371 |
| Climatology | 2.368 | 10.234 | 0.123 |
| Gradient boosting | 1.956 | 8.765 | 0.456 |
| LSTM | 1.789 | 8.123 | 0.543 |

The LSTM outperforms both naive baselines and the gradient-boosting model on every metric, and is
the only model with a clearly positive R² alongside gradient boosting persistence actually scores
worse than just predicting the mean (negative R²), which is itself a useful result: it shows the
raw event counts are noisy enough that recent history alone isn't a reliable guide without the
additional engineered features.

*Classification results (P(M >= 5.5 next month): PR-AUC and Brier skill score vs. climatology) 
add these from Section 6.1 of the notebook once you have them.*

---

## Limitations & next steps

- Deterministic prediction of a single earthquake's time, place, and magnitude is not attempted 
  it's an open problem in seismology. This forecasts *rates*, consistent with how operational
  systems like the USGS's aftershock forecasts and the academic ETAS model approach it.
- Each grid cell is modeled independently; seismicity is spatially coupled across neighboring
  cells, which a graph neural network over cell adjacency would capture better.
- No declustering (e.g. Gardner-Knopoff) is applied, so aftershock sequences and background
  seismicity are modeled together rather than separately.
- `predict_lstm.py`'s manual-entry mode reimplements the notebook's feature engineering
  independently rather than importing shared code fine for a demo, but a shared module would
  guarantee train/serve consistency in a real deployment.

---

## Data sources

- [USGS FDSN Event API](https://earthquake.usgs.gov/fdsnws/event/1/) - used here
- [ISC Bulletin](http://www.isc.ac.uk/iscbulletin/) - most complete global catalog, better pre-1990
- [Global CMT](https://www.globalcmt.org/) - focal mechanisms / moment tensors
- [SCEDC](https://scedc.caltech.edu/) - dense Southern California catalog
- [JMA](https://www.data.jma.go.jp/svd/eqev/data/bulletin/) - Japan catalog

## License

See [LICENSE](LICENSE).
