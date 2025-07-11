"""
Example of using the synthesize function to create a synthetic population.

This example demonstrates how to use the IPU algorithm to generate weights
and then use the synthesize function to create a synthetic population.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pyipu import ipu, synthesize

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
    print("\nHousehold size targets:")
    print(hh_targets['siz'])


    # Run IPU
    result = ipu(hh_seed, hh_targets, max_iterations=10, verbose=True)

    # Print results
    print("\nHousehold table with weights:")
    print(result['weight_tbl'])
    
    # Create a synthetic population
    print("\nCreating synthetic population...")
    synthetic_pop = synthesize(result['weight_tbl'], group_by='geo_cluster')
    
    print("\nSynthetic population (first few rows):")
    print(synthetic_pop.head())
    
    # Analyze the synthetic population
    print("\nSynthetic population size:", len(synthetic_pop))

    # Check that the synthetic population has the expected marginals
    # print("\nChecking marginals...") 
    # print("Target margins: ", hh_targets['siz'])
    # print("Synthetic pop margins: ", synthetic_pop['siz'].value_counts())
    
    

if __name__ == "__main__":
    main()
