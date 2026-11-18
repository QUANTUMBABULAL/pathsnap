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

RESULTS_PLACEHOLDER
