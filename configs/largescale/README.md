# Experiment Descriptions

These experiments mostly correspond to results in Figures 8 and 10. 

Folders:

- ```subnetonly/``` contains subnetwork finding experiments with frozen weights. These need to be run with a specific prune rate (1 - k)
- ```baselines/``` contains learned dense weight baseline configs
- ```sample/``` contains configs corresponding to the sample baseline from Zhou et al. Choose a ```--score-init-constant=<c>``` where c is lower when you want fewer weights remaining