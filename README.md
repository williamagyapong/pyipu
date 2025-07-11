# PyIPU: Python Implementation of Iterative Proportional Updating

PyIPU is a Python package that implements the Iterative Proportional Updating (IPU) algorithm proposed by Ye et al. (2009) in the paper "Methodology to match distributions of both household and person attributes in generation of synthetic populations". This implementation is based on the [ipfr](https://github.com/dkyleward/ipfr) R package.

## Overview

The IPU algorithm is a general case of iterative proportional fitting that can satisfy two disparate sets of marginals that do not agree on a single total. A common example is balancing population data using household- and person-level marginal controls for survey expansion or synthetic population creation.

**Key features**:
- Support for both household and person level constraints
- Handling of multiple geographies
- Configurable convergence criteria
- Detailed reporting of results
- Faster than traditional IPF

## Installation

```bash
pip install pyipu
```

Or install from source:

```bash
git clone https://github.com/williamagyapong/pyipu.git
cd pyipu
pip install -e .
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage

### Basic Example

```python
import pandas as pd
import numpy as np
from pyipu import ipu

# Create a simple household seed table
hh_seed = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'siz': [1, 2, 2, 1],
    'weight': [1, 1, 1, 1],
    'geo_cluster': [1, 1, 2, 2]
})

# Create household targets
hh_targets = {}
hh_targets['siz'] = pd.DataFrame({
    'geo_cluster': [1, 2],
    '1': [75, 100],
    '2': [25, 150]
})

# Run IPU
result = ipu(hh_seed, hh_targets, max_iterations=5)

# Access the results
print(result['weight_tbl'])  # Household table with weights
print(result['primary_comp'])  # Comparison of results to targets
result['weight_dist']  # Matplotlib figure showing weight distribution for diagnostics
```

### Household and Person Example

TODO: Include an example of using the PyIPU package with household and person level constraints to demonstrate how to use the IPU algorithm with both household and person level seed tables and targets, which is a common use case in population synthesis.



### Matrix Balancing Example

```python
import numpy as np
from pyipu import ipu_matrix

# Create a matrix
mtx = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Define row and column targets
row_targets = np.array([100, 200, 300])
col_targets = np.array([200, 250, 150])

# Balance the matrix
balanced_mtx = ipu_matrix(mtx, row_targets, col_targets)

print(balanced_mtx)
print("Row sums:", balanced_mtx.sum(axis=1))
print("Column sums:", balanced_mtx.sum(axis=0))
```

### Synthetic Population Generation Example 1

```python
import pandas as pd
from pyipu import ipu, synthesize

# Create a simple household seed table
hh_seed = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'siz': [1, 2, 2, 1],
    'income': ['low', 'med', 'high', 'low'],
    'weight': [1, 1, 1, 1],
    'geo_cluster': [1, 1, 2, 2]
})

# Create household targets
hh_targets = {}
hh_targets['siz'] = pd.DataFrame({
    'geo_cluster': [1, 2],
    '1': [75, 100],
    '2': [25, 150]
})

hh_targets['income'] = pd.DataFrame({
    'geo_cluster': [1, 2],
    'low': [60, 120],
    'med': [30, 80],
    'high': [10, 50]
})

# Run IPU
result = ipu(hh_seed, hh_targets, max_iterations=10)

# Create a synthetic population
synthetic_pop = synthesize(result['weight_tbl'], group_by='geo_cluster')

print("Synthetic population (first 10 rows):")
print(synthetic_pop.head(10))
```

## API Reference

### ipu

```python
ipu(primary_seed, primary_targets, 
    secondary_seed=None, secondary_targets=None,
    primary_id="id", secondary_importance=1,
    relative_gap=0.01, max_iterations=100, absolute_diff=10,
    weight_floor=0.00001, verbose=False,
    max_ratio=10000, min_ratio=0.0001)
```

**Parameters:**

- `primary_seed`: DataFrame containing the primary seed table (e.g., households)
- `primary_targets`: Dictionary of DataFrames with target marginals for primary seed
- `secondary_seed`: Optional DataFrame containing the secondary seed table (e.g., persons)
- `secondary_targets`: Optional dictionary of DataFrames with target marginals for secondary seed
- `primary_id`: Column name that links primary and secondary seed tables
- `secondary_importance`: Value between 0 and 1 signifying the importance of secondary targets
- `relative_gap`: Convergence threshold for percent RMSE between iterations
- `max_iterations`: Maximum number of iterations to perform
- `absolute_diff`: Threshold below which absolute differences don't matter for reporting
- `weight_floor`: Minimum weight to allow in any cell
- `verbose`: Whether to print iteration details and worst marginal stats
- `max_ratio`: Maximum weight as a multiple of the average weight
- `min_ratio`: Minimum weight as a multiple of the average weight

**Returns:**

A dictionary with the following keys:
- `weight_tbl`: The primary_seed with weight, avg_weight, and weight_factor columns
- `weight_dist`: A matplotlib figure showing the weight distribution
- `primary_comp`: A DataFrame comparing the primary seed results to targets
- `secondary_comp`: A DataFrame comparing the secondary seed results to targets (only if secondary_seed is provided)

### ipu_matrix

```python
ipu_matrix(mtx, row_targets, column_targets, **kwargs)
```

**Parameters:**

- `mtx`: 2D numpy array to balance
- `row_targets`: Array of targets for row sums
- `column_targets`: Array of targets for column sums
- `**kwargs`: Additional arguments passed to `ipu()`

**Returns:**

A 2D numpy array with the balanced matrix

### synthesize

```python
synthesize(weight_tbl, group_by=None, primary_id="id")
```

**Parameters:**

- `weight_tbl`: DataFrame containing the weight table output by `ipu()`
- `group_by`: Optional column name to group by before sampling (e.g., geography)
- `primary_id`: Column name of the primary ID in the weight table

**Returns:**

A DataFrame with one record for each synthesized member of the population. A `new_id` column is created, but the previous `primary_id` column is maintained to facilitate joining back to other data sources.

## License

MIT

## Citation

If you use PyIPU in your research, please cite:

```
- William, O. A. (2025). *PyIPU*: Python implementation of the Iterative 
Proportional Updating (IPU) algorithm [https://www.github.com/williamagyapong/pyipu]
(Version 0.1.0).

- Ye, X., Konduri, K., Pendyala, R. M., Sana, B., & Waddell, P. (2009). 
A methodology to match distributions of both household and person attributes 
in the generation of synthetic populations. 
In 88th Annual Meeting of the Transportation Research Board, Washington, DC.
```

## Package Structure

```
pyipu/
├── pyipu/
│   ├── __init__.py         # Package initialization
│   ├── version.py          # Version information
│   ├── core.py             # Main IPU implementation
│   ├── utils.py            # Helper functions
│   └── synthesis.py        # Synthetic population generation
├── examples/
│   ├── __init__.py
│   ├── basic_example.py    # Simple example with household data
│   ├── household_person_example.py  # Example with household and person data
│   ├── matrix_example.py   # Example of a simple matrix balancing
│   └── synthesis_example.py  # Example for synthetic population generation
├── tests/
│   ├── __init__.py
│   └── test_ipu.py         # Unit tests
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
├── README.md               # Documentation
├── run_tests.py            # Script to run tests
└── setup.py                # Package installation script
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgment

A big thanks to Kyle Ward, the author of the [ipfr](https://github.com/dkyleward/ipfr) package which provided the implementation logic for `pyipu`. It is fair to say that `pyipu` is the Python version of `ipfr`.
