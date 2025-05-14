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
        # 'income': ['low', 'med', 'high', 'low'],
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
    
    # hh_targets['income'] = pd.DataFrame({
    #     'geo_cluster': [1, 2],
    #     'low': [60, 120],
    #     'med': [30, 80],
    #     'high': [10, 50]
    # })

    print("Household seed table:")
    print(hh_seed)
    print("\nHousehold size targets:")
    print(hh_targets['siz'])
    # print("\nHousehold income targets:")
    # print(hh_targets['income'])

    # Run IPU
    result = ipu(hh_seed, hh_targets, max_iterations=10, verbose=True)

    # Print results
    print("\nHousehold table with weights:")
    print(result['weight_tbl'])
    
    print("\nComparison of results to targets:")
    print(result['primary_comp'])
    
    # Create a synthetic population
    print("\nCreating synthetic population...")
    synthetic_pop = synthesize(result['weight_tbl'], group_by='geo_cluster')
    
    print("\nSynthetic population (first few rows):")
    print(synthetic_pop.head())
    
    # Analyze the synthetic population
    print("\nSynthetic population size:", len(synthetic_pop))

    # Check that the synthetic population has the expected marginals
    print("\nChecking marginals...") 
    print("Target margins: ", hh_targets['siz'])
    print("Synthetic pop margins: ", synthetic_pop['siz'].value_counts())
    
    # Check distribution of size by geo_cluster
    size_dist = synthetic_pop.groupby(['geo_cluster', 'siz']).size().reset_index(name='count')
    size_dist_pivot = size_dist.pivot(index='geo_cluster', columns='siz', values='count').fillna(0)
    print("\nDistribution of household size by geo_cluster:")
    print(size_dist_pivot)
    
    # Check distribution of income by geo_cluster
    # income_dist = synthetic_pop.groupby(['geo_cluster', 'income']).size().reset_index(name='count')
    # income_dist_pivot = income_dist.pivot(index='geo_cluster', columns='income', values='count').fillna(0)
    # print("\nDistribution of household income by geo_cluster:")
    # print(income_dist_pivot)
    
    # # Compare with targets
    # print("\nComparing synthetic population with targets:")
    
    # # Size comparison
    # print("\nSize comparison:")
    # size_comparison = pd.DataFrame({
    #     'geo_cluster': [1, 2],
    #     'size_1_target': [75, 100],
    #     'size_1_synthetic': [size_dist_pivot.loc[1, 1] if 1 in size_dist_pivot.columns and 1 in size_dist_pivot.index else 0,
    #                         size_dist_pivot.loc[2, 1] if 1 in size_dist_pivot.columns and 2 in size_dist_pivot.index else 0],
    #     'size_2_target': [25, 150],
    #     'size_2_synthetic': [size_dist_pivot.loc[1, 2] if 2 in size_dist_pivot.columns and 1 in size_dist_pivot.index else 0,
    #                         size_dist_pivot.loc[2, 2] if 2 in size_dist_pivot.columns and 2 in size_dist_pivot.index else 0]
    # })
    # print(size_comparison)
    
    # # Income comparison
    # print("\nIncome comparison:")
    # income_comparison = pd.DataFrame({
    #     'geo_cluster': [1, 2],
    #     'low_target': [60, 120],
    #     'low_synthetic': [income_dist_pivot.loc[1, 'low'] if 'low' in income_dist_pivot.columns and 1 in income_dist_pivot.index else 0,
    #                      income_dist_pivot.loc[2, 'low'] if 'low' in income_dist_pivot.columns and 2 in income_dist_pivot.index else 0],
    #     'med_target': [30, 80],
    #     'med_synthetic': [income_dist_pivot.loc[1, 'med'] if 'med' in income_dist_pivot.columns and 1 in income_dist_pivot.index else 0,
    #                      income_dist_pivot.loc[2, 'med'] if 'med' in income_dist_pivot.columns and 2 in income_dist_pivot.index else 0],
    #     'high_target': [10, 50],
    #     'high_synthetic': [income_dist_pivot.loc[1, 'high'] if 'high' in income_dist_pivot.columns and 1 in income_dist_pivot.index else 0,
    #                       income_dist_pivot.loc[2, 'high'] if 'high' in income_dist_pivot.columns and 2 in income_dist_pivot.index else 0]
    # })
    # print(income_comparison)
    
    # # Visualize the results
    # # Size distribution
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # # Size distribution by geo_cluster
    # size_dist_pivot.plot(kind='bar', ax=ax1)
    # ax1.set_title('Household Size Distribution by Geo Cluster')
    # ax1.set_xlabel('Geo Cluster')
    # ax1.set_ylabel('Count')
    # ax1.legend(title='Household Size')
    
    # # Income distribution by geo_cluster
    # income_dist_pivot.plot(kind='bar', ax=ax2)
    # ax2.set_title('Household Income Distribution by Geo Cluster')
    # ax2.set_xlabel('Geo Cluster')
    # ax2.set_ylabel('Count')
    # ax2.legend(title='Household Income')
    
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
