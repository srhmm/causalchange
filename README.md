## CausalChange
*Causal Discovery under Changes in Distribution*: Implementations of the main algorithms included in the thesis.

### Setup
The following sets up a conda environment and a jupyter kernel (```Python (cc)```) that can be used to run the jupyter notebooks.
```
conda create -n causalchange python=3.10 -y
pip install -r requirements.txt
conda activate causalchange

pip install ipykernel
python -m ipykernel install --user --name causalchange --display-name "Python (cc)"
```

### Demos

The following jupyter notebooks under ``demo/`` show basic usage of the algorithms on example datasets.
- **Chapters 2, 3**: [TOPIC](demo/ch3_causal.ipynb) for causal DAG discovery from multiple contexts [1]

- **Chapter 5**: [CoCo](demo/ch4_confounding.ipynb) for counfounder detection from multiple contexts [2]
- **Chapter 5**: [SpaceTime](demo/ch5_time.ipynb) for TCG discovery and changepoint detection from time series [3]

- **Chapter 6**: [CMMs](demo/ch6_cmms.ipynb) for data from mixtures of populations [4]

Additional jupyter notebooks are available for the following case studies, 
 
- **Chapter 1**: [Population Paradox I](demo/ch1_example_simpson.ipynb), Simpson's paradox in COVID-19 data [5] 
- **Chapter 5**: [Changepoint Challenge](demo/ch5_realworld_river.ipynb), E-OBS temperature and precipitation data [6]
- **Chapter 6**: [Population Paradox III](demo/ch6_example_colon.ipynb), TCGA colon adenocarcinoma data [7]

### Experiments
To run larger experiments, we provide example experiment suites for the causal DAG discovery and causal CMM discovery,
- **Chapters 2, 3**: ```python -m src.exp.exp_change.exp_contexts ```
- **Chapter 6**: ```python -m src.exp.exp_change.exp_mix ```

For customization of experiment parameters or methods, see ```src/exp/exp_change/config_contexts.py```, respectively ```src/exp/exp_change/config_mix.py```.

### References

 - [1] Xu, S., Mameche, S., and Vreeken, J. *Information-theoretic causal discovery
in topological order.* AISTATS, 2025.
 - [2] Mameche, S., Vreeken, J., and Kaltenpoth, D. *Identifying confounding from
causal mechanism shifts.* AISTATS, 2024.
 - [3] Mameche, S., Cornanguer, L., Ninad, U., and Vreeken, J. *Spacetime: Causal
discovery from non-stationary time series.* AAAI, 2025a.
 - [4] Mameche, S., Kalofolias, J., and Vreeken, J. *Causal mixture models: Characterization and discovery.* NeurIPS, 2025b.
 - [5] based on data from Public Health England (2010), accessed at https://www.openintro.org/data/index.php?data=simpsons_paradox_covid
 - [6] based on E-OBS Temperature and Precipitation Data Sets, Cornes et al. (2018)
 - [7] based on Colon adenocarcinoma data, from the TCGA atlas, https://www.cancer.gov/tcga, as well as the RobMixReg package by Chang (2020) 



### Notes 

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