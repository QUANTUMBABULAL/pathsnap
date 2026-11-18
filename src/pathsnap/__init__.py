"""PathSnap: GPS map-matching and route reconstruction.

Modules
-------
geo         local metric projection for lon/lat data
io          raw trip readers (.track / .nodes / .arcs / .route)
network     road-network graph, arc geometry and local Dijkstra search
candidates  spatial candidate generation (GPS point -> nearby road arcs)
features    feature extraction for every (GPS point, candidate arc) pair
labels      reference-label derivation from the .route file (training / evaluation only)
hmm         HMM + Viterbi route reconstruction
pipeline    end-to-end inference used by the notebook and the API
evaluation  metrics (point-level and route-level)
"""

__version__ = "1.0.0"
