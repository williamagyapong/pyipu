"""
Advanced example of using the PyIPU package with household and person level constraints.

This example demonstrates how to use the IPU algorithm with both household and person
level seed tables and targets, which is a common use case in population synthesis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pyipu import ipu

def main():
    # Create household seed table
    hh_seed = pd.DataFrame({
        'id': [1, 2, 3],
        'hh_size': [1, 2, 3],
        'income': ['low', 'med', 'high'],
        'geo_taz': [1, 1, 2],
        'weight': [1, 1, 1]
    })

    # Create person seed table
    per_seed = pd.DataFrame({
        'id': [1, 2, 2, 3, 3, 3],
        'age': ['young', 'young', 'old', 'young', 'mid', 'old'],
        'gender': ['m', 'f', 'm', 'f', 'm', 'f']
    })

    # Create household targets
    hh_targets = {}
    hh_targets['hh_size'] = pd.DataFrame({
        'geo_taz': [1, 2],
        '1': [100, 50],
        '2': [200, 150],
        '3': [300, 400]
    })
    hh_targets['income'] = pd.DataFrame({
        'geo_taz': [1, 2],
        'low': [200, 150],
        'med': [250, 200],
        'high': [150, 250]
    })

    # Create person targets
    per_targets = {}
    per_targets['age'] = pd.DataFrame({
        'geo_taz': [1, 2],
        'young': [400, 350],
        'mid': [300, 400],
        'old': [200, 250]
    })
    per_targets['gender'] = pd.DataFrame({
        'geo_taz': [1, 2],
        'm': [450, 500],
        'f': [450, 500]
    })

    print("Household seed table:")
    print(hh_seed)
    print("\nPerson seed table:")
    print(per_seed)
    print("\nHousehold targets:")
    for name, target in hh_targets.items():
        print(f"\n{name}:")
        print(target)
    print("\nPerson targets:")
    for name, target in per_targets.items():
        print(f"\n{name}:")
        print(target)

    # Run IPU
    result = ipu(
        hh_seed, hh_targets,
        secondary_seed=per_seed, secondary_targets=per_targets,
        primary_id='id', secondary_importance=0.5,
        max_iterations=50, verbose=True
    )

    # Print results
    print("\nHousehold table with weights:")
    print(result['weight_tbl'])
    
    print("\nComparison of household results to targets:")
    print(result['primary_comp'])
    
    print("\nComparison of person results to targets:")
    print(result['secondary_comp'])
    

    # Display the weight distribution
    result['weight_dist']
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.hist(result['weight_tbl']['weight_factor'], bins=10, color='darkblue', edgecolor='gray')
    # plt.xlabel('Weight Ratio = Weight / Average Weight')
    # plt.ylabel('Count of Seed Records')
    # plt.title('Weight Distribution')
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
