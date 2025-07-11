"""
Synthesis functions for creating synthetic populations based on IPU results.
"""

import pandas as pd
import numpy as np


def synthesize(weight_tbl, group_by=None, primary_id="id"):
    """
    Create a synthetic population based on IPU results.
    
    A simple function that takes the weight_tbl output from ipu() and
    randomly samples based on the weight.
    
    Parameters
    ----------
    weight_tbl : pandas.DataFrame
        The DataFrame output by ipu() with the same name.
    group_by : str, optional
        If provided, the DataFrame will be grouped by this variable before
        sampling. If not provided, samples are drawn from the entire table.
    primary_id : str, default="id"
        The name of the primary ID column in the weight_tbl.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame with one record for each synthesized member of the
        population (e.g., household). A new_id column is created, but the
        previous primary_id column is maintained to facilitate joining back
        to other data sources (e.g., a person attribute table).
        
    Raises
    ------
    ValueError
        If primary_id or group_by is not found in weight_tbl.
        
    Examples
    --------
    >>> hh_seed = pd.DataFrame({
    ...     'id': [1, 2, 3, 4],
    ...     'siz': [1, 2, 2, 1],
    ...     'weight': [1, 1, 1, 1],
    ...     'geo_cluster': [1, 1, 2, 2]
    ... })
    >>> hh_targets = {}
    >>> hh_targets['siz'] = pd.DataFrame({
    ...     'geo_cluster': [1, 2],
    ...     '1': [75, 100],
    ...     '2': [25, 150]
    ... })
    >>> result = ipu(hh_seed, hh_targets, max_iterations=5)
    >>> synthetic_pop = synthesize(result['weight_tbl'], group_by='geo_cluster')
    """
    # Check if primary_id is in the weight_tbl
    if primary_id not in weight_tbl.columns:
        raise ValueError(f"primary_id '{primary_id}' not found in weight_tbl")
    
    # Make a copy to avoid modifying the original
    weight_tbl = weight_tbl.copy()
    
    # If group_by is provided, check if it's in the weight_tbl
    if group_by is not None:
        if group_by not in weight_tbl.columns:
            raise ValueError(f"group_by '{group_by}' not found in weight_tbl")
    
    # Calculate the total number of records to sample
    total_weight = weight_tbl['weight'].sum()
    n_samples = round(total_weight)
    
    # Initialize an empty DataFrame to store the results
    synthetic_table = pd.DataFrame()
    
    # If group_by is provided, sample within each group
    if group_by is not None:
        for group_val, group_df in weight_tbl.groupby(group_by):
            # Calculate the number of samples for this group
            group_weight = group_df['weight'].sum()
            group_samples = round(group_weight)
            # group_samples = group_weight # avoid rounding
            
            if group_samples > 0:
                # Sample from the group with replacement, weighted by 'weight'
                sampled_indices = np.random.choice(
                    group_df.index, 
                    size=group_samples, 
                    replace=True, 
                    p=group_df['weight'] / group_df['weight'].sum()
                )
                group_synthetic = group_df.loc[sampled_indices].copy()
                synthetic_table = pd.concat([synthetic_table, group_synthetic])
    else:
        # Sample from the entire table with replacement, weighted by 'weight'
        sampled_indices = np.random.choice(
            weight_tbl.index, 
            size=n_samples, 
            replace=True, 
            p=weight_tbl['weight'] / weight_tbl['weight'].sum()
        )
        synthetic_table = weight_tbl.loc[sampled_indices].copy()
        # synthetic_table = weight_tbl.sample(n_samples, replace=True, weights=weight_tbl['weight']).copy()
    # Drop weight-related columns
    cols_to_drop = ['weight', 'avg_weight', 'weight_factor']
    synthetic_table = synthetic_table.drop(columns=[col for col in cols_to_drop if col in synthetic_table.columns])
    
    # Add a new_id column
    synthetic_table['new_id'] = range(1, len(synthetic_table) + 1)
    
    # Reorder columns to put new_id and primary_id first
    cols = synthetic_table.columns.tolist()
    cols.remove('new_id')
    cols.remove(primary_id)
    synthetic_table = synthetic_table[['new_id', primary_id] + cols]
    
    # Reset the index
    synthetic_table = synthetic_table.reset_index(drop=True)
    
    return synthetic_table
