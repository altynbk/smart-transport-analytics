# Smart Transport Analytics

An intelligent system for predicting vehicle breakdowns from telematics and CAN data.
Diploma project, **Astana IT University** — School of Intelligent Systems, June 2026.

**Authors:** Auyen Yeginbay · Altynbek Kabiyev — group ST-2304, 6B06202 Smart Technology
**Scientific Advisor:** Assemgul Sadvakassova, Senior Lecturer

🔗 **Live demo:** https://altynbk.github.io/smart-transport-analytics/

---

## What it does

Four machine-learning subsystems for vehicle health, deployed entirely in the browser:

| # | Subsystem | Method | Result |
|---|-----------|--------|--------|
| 1 | **Scania APS** failure prediction | Cost-sensitive XGBoost | Total Cost 8,910 (rank 5/13), ROC AUC 0.9958 |
| 2 | **VED** driving-anomaly detection | Isolation Forest (unsupervised) | 22.4M readings, 384 vehicles |
| 3 | **OBD-II** real-time stress | XGBoost (3-class) | 99.73% accuracy, **22 µs** in the browser |
| 4 | **CAN-bus** intrusion detection | 4 × XGBoost (cross-file evaluation) | Honest measurement of a **46.7-point** data-leakage gap |

All models exported to ONNX, run client-side via ONNX Runtime Web (no server, no data leaves the device).

---

## How to run

### Live (recommended)
Open https://altynbk.github.io/smart-transport-analytics/ in any modern browser.
Tap the install button to add it as an app — works fully offline after that.

### Locally
```bash
# Serve the dashboard (any static server works)
python -m http.server 8000
# or VS Code "Go Live"
# Then open  http://localhost:8000
```

---

## Project structure

```
smart-transport-analytics/
├── index.html, app.js, styles.css     ← dashboard (root for clean Pages URL)
├── manifest.json, sw.js, icon.png     ← PWA (installable, works offline)
├── vendor/                            ← ONNX Runtime, Lucide, fonts (local)
├── assets/                            ← dashboard data, models (.onnx), plots
│
├── notebooks/                         ← training notebooks (01-05)
├── data/field/                        ← our own collected data
├── models/                            ← trained .onnx and .pkl
├── outputs/                           ← generated figures and results
└── docs/                              ← thesis PDF, presentation (optional)
```

---

## Datasets

### Our own data (included in this repo)

**`data/field/`** — **17,635 real OBD-II samples** collected by us from a vehicle
over two highway trips on **16 May 2026** and **17 May 2026**. Recorded via a
Vgate iCar Pro BLE 4.0 dongle. Used by notebook `05_field_validation.ipynb`
to validate the deployed OBD-II model on real driving data.

### Public datasets (download separately — not included)

These are large public datasets we used during training. Download them from
the original sources and place them in the paths shown below to reproduce
notebooks 01-04. The pre-computed outputs are already saved in each notebook,
so you can read everything without downloading.

| Dataset | Notebook | Source | Place in |
|---------|----------|--------|----------|
| Scania APS Failure | `01_scania_aps.ipynb` | https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks | `data/scania/` |
| VED (Vehicle Energy Dataset) | `02_ved_anomaly.ipynb` | https://github.com/gsoh/VED | `data/ved/` |
| OBD-II driving data | `03_obd_stress.ipynb` | https://www.kaggle.com/datasets/cabaki/obd-ii-driving-data | `data/obd2/` |
| HCRL Car-Hacking (CAN) | `04_can_intrusion.ipynb` | https://ocslab.hksecurity.net/Datasets/car-hacking-dataset | `data/can/` |

Each dataset is governed by its own licence.

---

## Key contributions

1. **Multi-domain integration** — failure, behaviour, stress and security in one deployable system.
2. **Honest measurement of data leakage** — the 46.7-point cross-file gap most CAN-IDS papers hide.
3. **Working browser deployment** — train once in Python, run anywhere via ONNX, validated end-to-end on a real car with our own collected data.

## License

This project is licensed under the MIT License — see `LICENSE`.
Third-party datasets are governed by their respective licenses (see "Datasets" above).
