# Data sources, provenance, and licensing

All datasets are small, inspectable, and either bundled with a library or downloaded once and **cached to
`data/`** by the loaders in `mvslib/data.py`. After the first run they work offline. No personal or sensitive
data is used.

| Dataset | Used in | Loader | Rows × cols | Source & provenance | Licence |
|---|---|---|---|---|---|
| **Wine recognition** | 03 PCA, 04 FA, 06 LDA, 08 | `data.load_wine_df()` | 178 × 13 (+target) | UCI ML Repository (Forina et al.); ships inside scikit-learn (`sklearn.datasets.load_wine`). Chemical analysis of wines from three cultivars in the same Italian region. | Public domain (UCI); BSD-3 via scikit-learn packaging |
| **Diabetes** | 02 Regression | `data.load_diabetes_df()` | 442 × 10 (+target) | Efron, Hastie, Johnstone & Tibshirani (2004), "Least Angle Regression"; ships inside scikit-learn (`load_diabetes`). Predictors are mean-centred / scaled at source. | Public (as distributed by scikit-learn, BSD-3) |
| **Palmer Penguins** | 05 Clustering, 06 Logistic, 07 Case study | `data.load_penguins_df()` | 344 × 7 (333 after dropna) | Horst AM, Hill AP, Gorman KB (2020), *palmerpenguins* R package. Collected by Dr. Kristen Gorman at Palmer Station LTER, Antarctica. Downloaded via `seaborn.load_dataset("penguins")` and cached. | **CC0 1.0** (public domain dedication) |
| **Iris** | 01 Linear algebra | `sklearn.datasets.load_iris` (inline) | 150 × 4 | Fisher (1936) / Anderson; ships inside scikit-learn. | Public domain |

## Notes

- **Network:** `load_penguins_df()` downloads the penguins CSV from the seaborn-data mirror on first use, then
  reads the local `data/penguins.csv` cache thereafter. Wine, Diabetes, and Iris are bundled with scikit-learn
  and require no network. To pre-cache everything, run any notebook that uses penguins once while online (or
  call `python -c "from mvslib import data; data.load_penguins_df()"`).
- **Caching:** the `data/` directory is created on demand. Deleting it just forces a re-download/rebuild.
- **Synthetic data:** several notebooks also generate small synthetic datasets with a fixed seed
  (`set_seed(7)`) where the mechanism needs to be visually obvious (e.g. the 2-factor model in 04, the blobs
  in 05). These have no licensing concerns.
- **Citation (penguins):** Gorman KB, Williams TD, Fraser WR (2014). *Ecological sexual dimorphism and
  environmental variability within a community of Antarctic penguins (genus Pygoscelis).* PLoS ONE 9(3):e90081.
