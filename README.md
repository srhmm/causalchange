## CausalChange
Implementations of the main algorithms included in the thesis *Causal Discovery under Distribution Change* (2025).

### Setup

#### conda
The following script sets up a conda environment and a jupyter kernel (```Python (cc)```) that can be used to run the jupyter notebooks,
```./scripts/conda_setup.sh```.

The following does the same,
```
conda create -n causalchange python=3.10 -y
pip install -r requirements.txt
conda activate causalchange

pip install ipykernel
python -m ipykernel install --user --name causalchange --display-name "Python (causc)"
```
#### docker
For a setup with Docker, after installing Docker (and making sure to
``` export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin/"``` if on mac), use
```./scripts/docker_setup.sh```.
 

### Demos

The following jupyter notebooks under ``demo/`` show basic usage of the algorithms on example datasets.
- **Chapters 2, 3**: [TOPIC](demo/ch3_causal.ipynb) for causal DAG discovery from multiple contexts [1, 2]

- **Chapter 5**: [CoCo](demo/ch4_confounding.ipynb) for detecting counfounding in multiple contexts [3]
- **Chapter 5**: [SpaceTime](demo/ch5_time.ipynb) for TCG discovery and changepoint detection in multi-context time series [4]

- **Chapter 6**: [CMMs](demo/ch6_cmms.ipynb) for causal modelling and discovery in mixtures of populations [5]

Additional jupyter notebooks are available for the following case studies, 
 
- **Chapter 1**: [Population Paradox I](demo/ch1_example_simpson.ipynb), Simpson's paradox in COVID-19 data [6] 
- **Chapter 5**: [Changepoint Challenge](demo/ch5_realworld_river.ipynb), E-OBS temperature and precipitation data [7]. The data is not public, but the notebooks show some geographical maps and changepoints discovered with SPACETIME.
- **Chapter 5**, for theFLUXNET data [9], the t-SNE embedding of SPACETIME's causal edge weights can be shown with this web app: ```python src\exp\exp_stime\app_tsne_dash.py``` with argument  ```--mode 0``` for different colorings (0-7)
- **Chapter 6**: [Population Paradox III](demo/ch6_example_colon.ipynb), TCGA colon adenocarcinoma data [8]

### Experiments
To run larger experiments, there are example experiment suites for the causal DAG discovery and causal CMM discovery,
- **Chapters 2, 3**: ```python -m src.exp.exp_change.exp_contexts ```
- **Chapter 6**: ```python -m src.exp.exp_change.exp_mix ```

For customization of experiment parameters or methods, see ```src/exp/exp_change/config_contexts.py```, respectively ```src/exp/exp_change/config_mix.py```.

### References

 - [1] Mameche, S., Kaltenpoth, D., and Vreeken, J. *Learning Causal Models under
Independent Changes.* NeurIPS, 2023.
 - [2] Xu, S., Mameche, S., and Vreeken, J. *Information-theoretic Causal Discovery
in Topological Order.* AISTATS, 2025.
 - [3] Mameche, S., Vreeken, J., and Kaltenpoth, D. *Identifying Confounding from
Causal Mechanism Shifts.* AISTATS, 2024.
 - [4] Mameche, S., Cornanguer, L., Ninad, U., and Vreeken, J. *Spacetime: Causal
Discovery from Non-stationary Time Series.* AAAI, 2025a.
 - [5] Mameche, S., Kalofolias, J., and Vreeken, J. *Causal Mixture Models: Characterization and Discovery.* NeurIPS, 2025b.
 - [6] based on data from Public Health England (2010), accessed at https://www.openintro.org/data/index.php?data=simpsons_paradox_covid
 - [7] based on E-OBS Temperature and Precipitation Data Sets, Cornes et al. (2018)
 - [8] based on Colon adenocarcinoma data, from the TCGA atlas, https://www.cancer.gov/tcga, as well as the RobMixReg package by Chang (2020) 
 - [8] based on FLUXNET data, obtained from https://fluxnet.org/data/, and based on Baldocchi, 2014 



### Notes 
To view the FLUXNET embedding, with
```set PYTHONPATH=%cd%```, 
```python -m src.exp.exp_stime.app_tsne_dash``` or ```python src\exp\exp_stime\app_tsne_dash.py```

*Note:* If the requirement of rpy is problematic, it can be omitted from the requirements.txt. Only the notebooks for **Chapter 6**
need it, the remaining notebooks can be used without it.

To install R packages, such as ```pcalg```, one can use the following.

``` 
from rpy2 import robjects as ro
ro.r('''
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(c("graph","RBGL"), ask=FALSE, update=FALSE)
# Optional (only if you want plotting via Rgraphviz; requires Graphviz installed on the OS):
# BiocManager::install("Rgraphviz", ask=FALSE, update=FALSE)
install.packages("pcalg", dependencies=TRUE)  
''') 
ro.r('suppressMessages(library(pcalg)); packageVersion("pcalg")') 
```