# AD-model(Anomaly-detection model) Developed and improved based on the GDformer

(GDformer: Going Beyond Subsequence Isolation for Multivariate Time Series Anomaly Detection)


## Get Started

1. Install >= Python 3.6, PyTorch >= 1.4.0.
2. Download data. You can obtain the benchmark datasets from the Github Repository of DCdetector ( [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR) ).
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/SWaT.sh
bash ./scripts/PSM.sh
```
