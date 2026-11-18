# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon · Team 9

PathSnap matches noisy GPS trajectories to a road network. It scores every (GPS point, nearby road arc) pair with a
trained classifier, then reconstructs a coherent route with an HMM decoded by Viterbi. The system runs as a FastAPI backend
(Laptop 1) and a browser UI (Laptop 2).

```
GPS track ─► cleaning ─► local metric projection ─► candidate arcs (KD-tree, r=50 m, k=8)
          ─► 22 features per (point, candidate) ─► classifier P(true arc)
          ─► HMM emissions + network-distance transitions ─► Viterbi ─► gap filling ─► route
```

## Repository layout

| Path | Contents |
|---|---|
| `data/train`, `data/student_test`, `data/evaluator_test` | raw trips (`.track`, `.nodes`, `.arcs`, `.route`) |
| `src/pathsnap/` | the library shared by the notebook and the API |
| &nbsp;&nbsp;`io.py` | raw readers and GPS cleaning (`.route` is only read by `load_route`) |
| &nbsp;&nbsp;`geo.py` | local equirectangular projection, point-to-segment geometry |
| &nbsp;&nbsp;`network.py` | directed road graph, reverse twins, bounded Dijkstra |
| &nbsp;&nbsp;`candidates.py` | densified KD-tree candidate generation |
| &nbsp;&nbsp;`features.py` | **feature extraction** (geometric, directional, kinematic, context-window, topology) |
| &nbsp;&nbsp;`labels.py` | monotone DP alignment of reference routes to GPS points (training and scoring only) |
| &nbsp;&nbsp;`hmm.py` | HMM + Viterbi and route reconstruction |
| &nbsp;&nbsp;`pipeline.py` | end-to-end inference plus artifact save/load |
| &nbsp;&nbsp;`evaluation.py` | candidate, point and route metrics |
| &nbsp;&nbsp;`viz.py` | Folium map generation |
| `models/PathSnap_Training.ipynb` | **all training**: EDA, features, 5 models, Bayesian HPO, HMM, maps, export |
| `models/artifacts/` | `final_model.joblib` and `pipeline_config.json` (loaded by the API) |
| `models/results/` | every experiment table as CSV, plus the Optuna trial log |
| `models/figures/`, `models/maps/` | plots and interactive Folium maps |
| `api/` | FastAPI backend (`main.py`) and the server-side scorer (`scorer.py`) |
| `ui/` | evaluator UI (static HTML/JS, Leaflet) and `serve.py` |
| `docs/` | hackathon brief and guidelines |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1 · Train (reproduces every result and artifact)

```powershell
cd models
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 PathSnap_Training.ipynb
```

You can also open the notebook in Jupyter or VS Code and run all cells. The notebook never reads `data/evaluator_test`.

## 2 · Laptop 1: start the API

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

* `GET /health`: model and decoder configuration
* `GET /trips`: the 10 reserved evaluation trips (IDs and GPS stats only)
* `POST /predict` with body `{"trip_id": "00000015"}`: runs the full pipeline, then scores the result on the server

Every request, and the metrics for every prediction, is logged to the console and to `api/logs/api.log`. Interactive docs are at
`http://<laptop1-ip>:8000/docs`. If Windows Firewall prompts, allow Python on private networks.

## 3 · Laptop 2: start the UI

```powershell
python ui/serve.py --port 5500
```

Open `http://localhost:5500`, enter Laptop 1's address (for example `http://192.168.1.20:8000`) in **API connection**, and click
**Connect**. Pick one of the 10 trips and click **Run map-matching**. The map shows the raw GPS, the nearby road network, the
predicted route and the snapped points, with the metrics and latency breakdown on the right.

## Leakage safeguards

* The evaluator trips are used only by the API at demo time. The notebook, feature selection and tuning never read them.
* The prediction pipeline (`pipeline.py`) never reads `.route`. Only `api/scorer.py` reads it, after prediction.
* The API returns the predicted route and aggregate metrics only. It never sends reference geometry or per-point labels to the UI.
* Nothing is hardcoded: each request recomputes candidates, features, probabilities and Viterbi from the raw files.

## Results

All numbers below come from `models/results/*.csv`, written by the executed notebook.

### Model comparison (5-fold GroupKFold on 80 training trips; student-test = 10 held-out trips)

| Model | Family | CV macro-F1 | CV Top-1 | Student-test Top-1 | Student-test macro-F1 | Student-test log loss | Continuity (classifier only) | Inference (ms / 1k cand.) |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** | Gradient boosting | **0.9362 ± 0.012** | **0.9003** | **0.9060** | **0.9356** | **0.0845** | **0.806** | 3.2 |
| Random Forest | Bagging | 0.9356 ± 0.013 | 0.8964 | 0.8839 | 0.9298 | 0.0902 | 0.788 | 2.2 |
| MLP (64-32) | Neural network | 0.9317 ± 0.013 | 0.8952 | 0.8971 | 0.9338 | 0.0915 | 0.802 | 0.3 |
| KNN (k=25) | Instance-based | 0.9184 ± 0.014 | 0.8780 | 0.8683 | 0.9135 | 0.1496 | 0.775 | 78.7 |
| Logistic Regression | Linear | 0.9082 ± 0.008 | 0.8868 | 0.8705 | 0.9061 | 0.1320 | 0.746 | 0.1 |

The selected model is **LightGBM**. It has the best CV macro-F1, which is the primary metric, and the best log loss, which matters because its probabilities become the HMM emissions.

### Bayesian optimisation (Optuna TPE, 30 trials, 3-fold group CV, objective = macro-F1)

| | Untuned | Tuned |
|---|---|---|
| CV macro-F1 | 0.9362 | **0.9370** |
| CV Top-1 | 0.9003 | 0.9010 |
| Student-test Top-1 | 0.9060 | 0.9039 |
| Student-test macro-F1 | 0.9356 | 0.9361 |
| Inference ms / 1k | 3.4 | 10.4 |

Tuning improved CV only marginally (the default configuration was already close to optimal) and roughly tripled inference
cost. Best parameters: 600 trees, learning rate 0.033, 182 leaves, min_child_samples 95.

### Feature-group ablation (3-fold group CV Top-1, LightGBM)

All features 0.904 · without directional 0.679 · without relative-geometry 0.855 · without context window 0.890 ·
without topology 0.897 · without kinematic 0.903 · distance only 0.572

### HMM / Viterbi (student-test, 19,160 GPS points)

| | Classifier only | Classifier + HMM |
|---|---|---|
| Observation Top-1 | 0.9039 | **0.9396** |
| Continuity of consecutive matches | 0.806 | **0.909** |
| Route F1 (length-weighted) | 0.852 | **0.976** |
| Reconstructed route continuity | – | 0.999 |

HMM decoding improved Top-1 on all 10 student-test trips. Decoder parameters (α = 0.5, β = 8) were chosen on training trips
using out-of-fold emissions.
