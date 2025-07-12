"""
Tests for the PyIPU package.
"""

import unittest
import numpy as np
import pandas as pd
from pyipu import ipu, ipu_matrix, synthesize


class TestIPU(unittest.TestCase):
    """Test cases for the IPU algorithm."""

    def test_basic_ipu(self):
        """Test basic IPU functionality with a simple household example."""
        # Create a simple household seed table
        hh_seed = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'siz': [1, 2, 2, 1],
            # 'weight': [1, 1, 1, 1], # weights of ones will be assigned by default
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
        result = ipu(hh_seed, hh_targets, max_iterations=10)

        # Check that the result contains the expected keys
        self.assertIn('weight_tbl', result)
        self.assertIn('weight_dist', result)
        self.assertIn('primary_comp', result)

        # Check that the weights sum to the expected totals
        weight_tbl = result['weight_tbl']
        
        # Calculate the sum of weights for size=1 in geo_cluster=1
        sum_1_1 = weight_tbl.loc[(weight_tbl['siz'] == 1) & 
                                 (weight_tbl['geo_cluster'] == 1), 'weight'].sum()
        
        # Calculate the sum of weights for size=2 in geo_cluster=1
        sum_2_1 = weight_tbl.loc[(weight_tbl['siz'] == 2) & 
                                 (weight_tbl['geo_cluster'] == 1), 'weight'].sum()
        
        # Calculate the sum of weights for size=1 in geo_cluster=2
        sum_1_2 = weight_tbl.loc[(weight_tbl['siz'] == 1) & 
                                 (weight_tbl['geo_cluster'] == 2), 'weight'].sum()
        
        # Calculate the sum of weights for size=2 in geo_cluster=2
        sum_2_2 = weight_tbl.loc[(weight_tbl['siz'] == 2) & 
                                 (weight_tbl['geo_cluster'] == 2), 'weight'].sum()
        
        # Check that the sums are close to the targets
        self.assertAlmostEqual(sum_1_1, 75, delta=1)
        self.assertAlmostEqual(sum_2_1, 25, delta=1)
        self.assertAlmostEqual(sum_1_2, 100, delta=1)
        self.assertAlmostEqual(sum_2_2, 150, delta=1)

    # TODO: Fix errors with this use case
    # def test_household_person_ipu(self):
    #     """Test IPU with both household and person level constraints."""
    #     # Create household seed table
    #     hh_seed = pd.DataFrame({
    #         'id': [1, 2, 3],
    #         'hh_size': [1, 2, 3],
    #         'geo_taz': [1, 1, 2],
    #         'weight': [1, 1, 1]
    #     })

    #     # Create person seed table
    #     per_seed = pd.DataFrame({
    #         'id': [1, 2, 2, 3, 3, 3],
    #         'age': ['young', 'young', 'old', 'young', 'mid', 'old']
    #     })

    #     # Create household targets
    #     hh_targets = {}
    #     hh_targets['hh_size'] = pd.DataFrame({
    #         'geo_taz': [1, 2],
    #         '1': [100, 50],
    #         '2': [200, 0],
    #         '3': [0, 400]
    #     })

    #     # Create person targets
    #     per_targets = {}
    #     per_targets['age'] = pd.DataFrame({
    #         'geo_taz': [1, 2],
    #         'young': [400, 350],
    #         'mid': [0, 400],
    #         'old': [200, 250]
    #     })

    #     # Run IPU
    #     result = ipu(
    #         hh_seed, hh_targets,
    #         secondary_seed=per_seed, secondary_targets=per_targets,
    #         primary_id='id', secondary_importance=0.5,
    #         max_iterations=50
    #     )

    #     # Check that the result contains the expected keys
    #     self.assertIn('weight_tbl', result)
    #     self.assertIn('weight_dist', result)
    #     self.assertIn('primary_comp', result)
    #     self.assertIn('secondary_comp', result)

    #     # Check that the household weights sum to the expected totals
    #     weight_tbl = result['weight_tbl']
        
    #     # Calculate the sum of weights for hh_size=1 in geo_taz=1
    #     sum_1_1 = weight_tbl.loc[(weight_tbl['hh_size'] == 1) & 
    #                              (weight_tbl['geo_taz'] == 1), 'weight'].sum()
        
    #     # Calculate the sum of weights for hh_size=2 in geo_taz=1
    #     sum_2_1 = weight_tbl.loc[(weight_tbl['hh_size'] == 2) & 
    #                              (weight_tbl['geo_taz'] == 1), 'weight'].sum()
        
    #     # Calculate the sum of weights for hh_size=3 in geo_taz=2
    #     sum_3_2 = weight_tbl.loc[(weight_tbl['hh_size'] == 3) & 
    #                              (weight_tbl['geo_taz'] == 2), 'weight'].sum()
        
    #     # Check that the sums are close to the targets
    #     self.assertAlmostEqual(sum_1_1, 100, delta=1)
    #     self.assertAlmostEqual(sum_2_1, 200, delta=1)
    #     self.assertAlmostEqual(sum_3_2, 400, delta=1)

    def test_matrix_ipu(self):
        """Test matrix balancing functionality."""
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
        
        # Check that the row and column sums match the targets
        row_sums = balanced_mtx.sum(axis=1)
        col_sums = balanced_mtx.sum(axis=0)
        
        for i, target in enumerate(row_targets):
            self.assertAlmostEqual(row_sums[i], target, delta=0.1)
            
        for i, target in enumerate(col_targets):
            self.assertAlmostEqual(col_sums[i], target, delta=0.1)
            
    def test_synthesize(self):
        """Test synthetic population generation functionality."""
        # Create a simple household seed table
        hh_seed = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'siz': [1, 2, 2, 1],
            # 'weight': [1, 1, 1, 1],
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
        result = ipu(hh_seed, hh_targets, max_iterations=10)
        
        # Create a synthetic population
        synthetic_pop = synthesize(result['weight_tbl'], group_by='geo_cluster')
        
        # Check that the synthetic population has the expected structure
        self.assertIn('new_id', synthetic_pop.columns)
        self.assertIn('id', synthetic_pop.columns)
        self.assertIn('siz', synthetic_pop.columns)
        self.assertIn('geo_cluster', synthetic_pop.columns)
        
        # Check that weight-related columns are removed
        self.assertNotIn('weight', synthetic_pop.columns)
        self.assertNotIn('avg_weight', synthetic_pop.columns)
        self.assertNotIn('weight_factor', synthetic_pop.columns)
        
        # Check that the total population size is approximately equal to the sum of weights
        total_weight = result['weight_tbl']['weight'].sum()
        self.assertAlmostEqual(len(synthetic_pop), total_weight, delta=1)
        
        # Check that the distribution by geography is approximately correct
        geo_1_weight = result['weight_tbl'].loc[result['weight_tbl']['geo_cluster'] == 1, 'weight'].sum()
        geo_2_weight = result['weight_tbl'].loc[result['weight_tbl']['geo_cluster'] == 2, 'weight'].sum()
        
        geo_1_count = len(synthetic_pop[synthetic_pop['geo_cluster'] == 1])
        geo_2_count = len(synthetic_pop[synthetic_pop['geo_cluster'] == 2])
        
        self.assertAlmostEqual(geo_1_count, geo_1_weight, delta=1)
        self.assertAlmostEqual(geo_2_count, geo_2_weight, delta=1)
        
        # Test with no group_by
        synthetic_pop2 = synthesize(result['weight_tbl'])
        self.assertEqual(len(synthetic_pop2), round(total_weight))
        
        # Test with different primary_id
        # First rename the id column
        weight_tbl_copy = result['weight_tbl'].copy()
        weight_tbl_copy = weight_tbl_copy.rename(columns={'id': 'household_id'})
        
        synthetic_pop3 = synthesize(weight_tbl_copy, primary_id='household_id')
        self.assertIn('new_id', synthetic_pop3.columns)
        self.assertIn('household_id', synthetic_pop3.columns)
        self.assertEqual(len(synthetic_pop3), round(total_weight))


if __name__ == '__main__':
    unittest.main()
