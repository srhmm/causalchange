### Architecture

---

structure of `causalchange`

```text
Topic / Linc / SpaceTime wrappers
--> CausalChange (general API)
--> config (input params)
--> engine factory (workflow high-level)
--> discovery engine (workflow for each algo)
--> component discovery (component of each algo)
--> result
```

responsibilities

```text
core/       - for shared types, protocols & results
config/     - for storing and checking input parameters
domain/     - for data checks and preprocessing for each data type (tabular, temporal, multicontext)
factory     - factory that builds discovery engine from hyperparameters
engines/    - discovery engines that run the algorithmic steps (to build TOPIC, LINC, SpaceTime algos and variations thereof)
discovery/  - algorithmic steps for each discovered object (graph search, changepoint detection, etc)
scoring/    - utilities for regression, statistical testing and caching thereof here
posthoc/    - utilities for further analysis after main algorithm was run
plotting/   - utilities for visuals
```
