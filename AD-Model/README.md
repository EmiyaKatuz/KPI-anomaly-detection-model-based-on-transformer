# AD-model(Anomaly-detection model) Developed and improved based on the GDformer

(GDformer: Going Beyond Subsequence Isolation for Multivariate Time Series Anomaly Detection)


## 1. Get Started--Environment configuration

 Install >= Python 3.6, PyTorch >= 1.4.0.

## 2.Datasets used
  Download data. You can obtain the benchmark datasets from the Github Repository of DCdetector ( [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR) ).


## 3.Experiment results reproduce 
4. Train and Test. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:

```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/SWaT.sh
bash ./scripts/PSM.sh
```


## 4.Experiment results

###4.1 Implementation Details



###4.2 Evaluation Metrics



###4.3 Baseline model


###4.4 Mian result


###4.5.1 Parameter Sensitivity


###4.5.2 Ablation Experiments


###4.5.3 Model efficiency




```bash
bash ./extra_test.sh
bash ./len_map.sh
bash ./num_proto.sh
```

........
