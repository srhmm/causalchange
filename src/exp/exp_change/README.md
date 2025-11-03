  
### Parameters
modify in ```config.py```
### Experiments

```
run.py
│   each exp config: { 'GEN': 'linear', 'NS': 'normal' }
│   exps = options.get_experiments()
│
└───run_cases.py
│   │   each exp case: { N=10, NZ=2, S=500 }
│   │   cases = options.get_cases() using attributes in exp_mixture_defaults()
│   │
│   └───run_case.py
│       │   each seed: i, each method: m
│       │   runs experiment of type options.exp_type
│       │   returns results, results[m] : SimpleNamespace('mth'=m, 'metrics'={'f1': ... })
│       └─   
└─ 


run_case.py
│
└───run_case_clustering.py
│   │   ExpType 0: evaluate mixing metrics (GZ)
│   │   samples a DAG G and structure GZ (exp, case, i)
│   │   evaluates mixing metrics (AMI for each Xj in G; F1(target); Jaccard(target sets))
│   └─   
└───run_case_power_specificity.py
│   │   ExpType 1: power, specificity improvements
│   │   samples a DAG G and structure GZ (exp, case, i)
│   │   compares causal discovery w,w/o confounders (power/type II errors), w,w/o causal mixtures (specificity/type I errors) 
│   └─  
└───run_case_causal_discovery
│   │   ExpType 2: causal discovery
│   │   samples a DAG G and structure GZ (exp, case, i)
│   │   evaluates G-hat (graph metrics, mixing metrics)
│   └─   
└─── run_case_interventional_mixture -- similar with NZ=1 
│  
└─
