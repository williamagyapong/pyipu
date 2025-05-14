"""
Basic example of using the PyIPU package.

This example demonstrates how to use the IPU algorithm with a simple household seed table
and household-level targets.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pyipu import ipu

def main():
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

    print("Household seed table:")
    print(hh_seed)
    print("\nHousehold targets:")
    print(hh_targets['siz'])

    # Run IPU
    result = ipu(hh_seed, hh_targets, max_iterations=10, verbose=True)
    # result = result['weight_tbl']
    # print(result.columns)
    # print(type(result))
    # print(result.head())

    # Print results
    print("\nHousehold table with weights:")
    print(result['weight_tbl'])
    
    print("\nComparison of results to targets:")
    print(result['primary_comp'])

    
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
