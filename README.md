<div align="center">

# 📊 Projects

### *Exploratory data analysis, end to end.*

**A portfolio of data science projects — each one takes a raw public dataset and
works it through cleaning, analysis, and visualisation to a set of findings.**

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />

</div>

---

## The projects

| Project | Dataset | Focus | Notebook |
|---|---|---|:--:|
| [**UFO Sightings**](#-ufo-sightings) | 141,359 NUFORC reports | Large-scale cleaning, geospatial mapping, text analysis | 116 cells |
| [**US Arrests 1979**](#-us-arrests-1979) | Arrest rates across US states | PCA and clustering — unsupervised learning | 75 cells |
| [**Mental Health in Tech**](#-mental-health-in-tech-2014) | 2014 worldwide survey | Survey analysis, attitudes and frequency | 106 cells |
| [**Automobiles**](#-automobiles) | Vehicle specifications | Feature relationships and price drivers | 87 cells |

Plus a [**tech roadmaps**](#-tech-roadmaps) reference collection.

---

## 🛸 UFO Sightings

**[`[EDA]UFO Sightings/`](%5BEDA%5DUFO%20Sightings)** · [notebook](%5BEDA%5DUFO%20Sightings/UFO_Sightings.ipynb)

Eyewitness UFO reports collected by the National UFO Research Center — **141,359 rows
across 13 columns**, from a 412MB source CSV.

The largest project here, and the one that's really about **data cleaning at scale**:
handling missing values, malformed records, and free-text fields across a dataset too
big to eyeball. From there it moves into where sightings cluster geographically, which
shapes get reported, and how sightings distribute over time.

**Techniques** — data cleaning, geospatial mapping with `geopandas` and shapefiles,
natural language processing on report text, word clouds, interactive Plotly charts.

## 🚔 US Arrests 1979

**[`[EDA]_US_ARRESTS-main/`](%5BEDA%5D_US_ARRESTS-main)** · [notebook](%5BEDA%5D_US_ARRESTS-main/US_Arrests_1979_survey.ipynb)

Arrest statistics per 100,000 residents across US states, broken down by murder,
assault, and rape.

The **unsupervised learning** project. Rather than predicting a target, it asks what
structure the data has on its own: principal component analysis to find the directions
the states differ along most, then clustering to group states by arrest profile.

**Techniques** — PCA, clustering, `scikit-learn`, statistical analysis with `scipy`.

## 🧠 Mental Health in Tech (2014)

**[`[EDA]-Mental_Health_Survey_2014/`](%5BEDA%5D-Mental_Health_Survey_2014-main)** · [notebook](%5BEDA%5D-Mental_Health_Survey_2014-main/Mental_Health_Tech.ipynb)

A worldwide 2014 survey measuring attitudes toward mental health, and the frequency of
mental health conditions, inside tech companies.

Survey data brings its own problems — free-text answers, inconsistent categories,
self-selection. The analysis works through those to look at how attitudes vary by
country, employer, and workplace policy, and where the gaps between stated policy and
lived experience appear.

**Techniques** — survey data cleaning, association rule mining with `mlxtend`,
categorical analysis, word clouds.

## 🚗 Automobiles

**[`[EDA]Automobiles/`](%5BEDA%5DAutomobiles)** · [notebook](%5BEDA%5DAutomobiles/Automobiles_project.ipynb)

Vehicle specifications across manufacturers — engine, body, and performance
attributes alongside price.

A focused look at **what actually drives selling price**: which specifications
correlate with it, which are noise, and how manufacturers differ once you control for
the obvious factors.

**Techniques** — inferential statistics, missing-data profiling with `missingno`,
correlation analysis, visualisation.

## 🗺 Tech Roadmaps

**[`Roadmaps_Tech-main/`](Roadmaps_Tech-main)**

A reference set of career and skill roadmaps — Data Analyst, Generative AI, NLP,
DevOps, ML Algorithms, Tech Stacks, and Tech Recruiter.

> Compiled from [roadmap.sh](https://roadmap.sh/) and the community-built
> [kamranahmedse/developer-roadmap](https://github.com/kamranahmedse/developer-roadmap)
> project. All credit to their authors — these are collected here for reference, not
> authored by me.

---

## Data sources

Every dataset here is public, and each project's own README records where it came
from. In summary:

| Project | Dataset | Source |
|---|---|---|
| UFO Sightings | `nuforc_reports.csv` | Timothy Renner — public domain |
| US Arrests 1979 | `USArrests.csv` | Open source, author unknown |
| Mental Health in Tech | `survey.csv` (renamed `mentalhealth.csv`) | Stephen Myers |
| Automobiles | `automobile.txt` | Supplied by HyperionDev; original source unknown |

Datasets remain the property of their respective authors. The analysis and notebooks
are mine.

---

## Running the notebooks

Each project folder is self-contained. The notebooks have their outputs saved, so
they're readable on GitHub without running anything.

To run one yourself:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy
jupyter notebook
```

Individual projects need extras — `geopandas`, `fiona` and `nltk` for UFO Sightings,
`mlxtend` and `wordcloud` for Mental Health, `missingno` for Automobiles.

> **Note:** the UFO Sightings source CSV is 412MB and is **not** committed here. The
> notebook's saved outputs show the analysis; re-running it from scratch means
> fetching the dataset from its original source.

---

## Related repositories

- [**eda-toolkit**](https://github.com/warsab/eda-toolkit) — the helper library that
  came out of doing this kind of analysis repeatedly
- [**ml-algorithms**](https://github.com/warsab/ml-algorithms) — 24 ML algorithm
  templates, each explained and runnable
- [**cheatsheets**](https://github.com/warsab/cheatsheets) — technical reference sheets

---

<div align="center">
<sub>Built by Warrick Sabatta · Feel free to use, share, and adapt these projects.</sub>
</div>
