"""
Core implementation of the Iterative Proportional Updating (IPU) algorithm.

This module contains the main IPU function and related functionality for
balancing population data using household- and person-level marginal controls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from .utils import (
    check_tables, scale_targets, balance_secondary_targets,
    process_seed_table, compare_results
)


def ipu(primary_seed, primary_targets, 
        secondary_seed=None, secondary_targets=None,
        primary_id="id", secondary_importance=1,
        relative_gap=0.01, max_iterations=100, absolute_diff=10,
        weight_floor=0.00001, verbose=False,
        max_ratio=10000, min_ratio=0.0001):
    """
    Iterative Proportional Updating algorithm.
    
    A general case of iterative proportional fitting. It can satisfy
    two, disparate sets of marginals that do not agree on a single total. A
    common example is balancing population data using household- and person-level
    marginal controls. This could be for survey expansion or synthetic
    population creation. The second set of marginal/seed data is optional, meaning
    it can also be used for more basic IPF tasks.
    
    Parameters
    ----------
    primary_seed : pandas.DataFrame
        In population synthesis or household survey expansion, 
        this would be the household seed table (each record would represent a 
        household). It could also be a trip table, where each row represents an 
        origin-destination pair.
    primary_targets : dict
        A dictionary of DataFrames. Each key in the dictionary defines a 
        marginal dimension and must match a column from the primary_seed table. 
        The DataFrame associated with each key can contain a geography field 
        (starting with "geo_"). If so, each row in the target table defines a 
        new geography (these could be TAZs, tracts, clusters, etc.). The other 
        column names define the marginal categories that targets are provided for.
    secondary_seed : pandas.DataFrame, optional
        Most commonly, if the primary_seed describes households, the secondary 
        seed table would describe the persons in each household. Must contain 
        the same primary_id column that links each person to their respective 
        household in primary_seed.
    secondary_targets : dict, optional
        Same format as primary_targets, but they constrain the secondary_seed table.
    primary_id : str, default="id"
        The field used to join the primary and secondary seed tables. Only 
        necessary if secondary_seed is provided.
    secondary_importance : float, default=1
        A value between 0 and 1 signifying the importance of the secondary targets. 
        At an importance of 1, the function will try to match the secondary targets 
        exactly. At 0, only the percentage distributions are used.
    relative_gap : float, default=0.01
        After each iteration, the weights are compared to the previous weights 
        and the %RMSE is calculated. If the %RMSE is less than the relative_gap 
        threshold, then the process terminates.
    max_iterations : int, default=100
        Maximum number of iterations to perform, even if relative_gap is not reached.
    absolute_diff : float, default=10
        Upon completion, the function will report the worst-performing marginal 
        category and geography based on the percent difference from the target. 
        absolute_diff is a threshold below which percent differences don't matter.
        
        For example, if a target value was 2, and the expanded weights equaled 1, 
        that's a 100% difference, but is not important because the absolute value
        is only 1.
    weight_floor : float, default=0.00001
        Minimum weight to allow in any cell to prevent zero weights. Should be 
        arbitrarily small compared to your seed table weights.
    verbose : bool, default=False
        Print iteration details and worst marginal stats upon completion?
    max_ratio : float, default=10000
        The average weight per seed record is calculated by dividing the total 
        of the targets by the number of records. The max_ratio caps the maximum 
        weight at a multiple of that average.
    min_ratio : float, default=0.0001
        The average weight per seed record is calculated by dividing the total 
        of the targets by the number of records. The min_ratio caps the minimum 
        weight at a multiple of that average.
        
    Returns
    -------
    dict
        A dictionary with the following keys:
        - weight_tbl: The primary_seed with weight, avg_weight, and weight_factor columns
        - weight_dist: A matplotlib figure showing the weight distribution
        - primary_comp: A DataFrame comparing the primary seed results to targets
        - secondary_comp: A DataFrame comparing the secondary seed results to targets
          (only if secondary_seed is provided)
    
    References
    ----------
    Ye, X., Konduri, K., Pendyala, R. M., Sana, B., & Waddell, P. (2009). 
    A methodology to match distributions of both household and person attributes 
    in the generation of synthetic populations. In 88th Annual Meeting of the 
    Transportation Research Board, Washington, DC.
    """
    # Check for valid values of secondary_importance
    if secondary_importance > 1 or secondary_importance < 0:
        raise ValueError("`secondary_importance` argument must be between 0 and 1")
    
    # Check hh and person tables
    if secondary_seed is not None:
        result = check_tables(
            primary_seed, primary_targets, primary_id=primary_id,
            secondary_seed=secondary_seed, secondary_targets=secondary_targets
        )
    else:
        result = check_tables(
            primary_seed, primary_targets, primary_id=primary_id
        )
    
    primary_seed, primary_targets, secondary_seed, secondary_targets = result
    
    # Scale target tables
    # All tables in the list will match the totals of the first table
    primary_targets = scale_targets(primary_targets, verbose)
    if secondary_seed is not None:
        secondary_targets = scale_targets(secondary_targets, verbose)
    
    # Balance secondary targets to primary
    if secondary_importance != 1 and secondary_seed is not None:
        if verbose:
            print("Balancing secondary targets to primary")
        secondary_targets_mod = balance_secondary_targets(
            primary_targets, primary_seed, secondary_targets, secondary_seed,
            secondary_importance, primary_id
        )
    else:
        secondary_targets_mod = secondary_targets
    
    # Pull off the geo information into a separate equivalency table
    # to be used as needed
    geo_cols = [col for col in primary_seed.columns if col.startswith("geo_")]
    geo_equiv = primary_seed[[primary_id, "weight"] + geo_cols].copy()
    
    # Process the seed table into dummy variables (one-hot encoding)
    marginal_columns = list(primary_targets.keys())
    primary_seed_mod = process_seed_table(
        primary_seed, primary_id, marginal_columns
    )
    
    if secondary_seed is not None:
        # Modify the person seed table the same way, but sum by primary ID
        marginal_columns = list(secondary_targets_mod.keys())
        secondary_seed_mod = process_seed_table(
            secondary_seed, primary_id, marginal_columns
        )
        
        # Group by primary_id to get household-level summaries
        secondary_seed_mod = secondary_seed_mod.groupby(primary_id).sum().reset_index()
        
        # Combine the hh and per seed tables into a single table
        seed = pd.merge(primary_seed_mod, secondary_seed_mod, on=primary_id, how="left")
    else:
        seed = primary_seed_mod
    
    # Add the geo information back
    seed = pd.merge(seed, geo_equiv, on=primary_id)

    # Ensure 'weight' column is float type to avoid dtype incompatibility
    seed["weight"] = seed["weight"].astype(float)
    
    # Store a vector of attribute column names to loop over later
    # Don't include primary_id, geo columns, or 'weight' in the vector
    seed_cols = seed.columns.tolist()
    geo_cols = [col for col in seed_cols if col.startswith("geo_")]
    seed_attribute_cols = [col for col in seed_cols 
                          if col != primary_id and col != "weight" and col not in geo_cols]
    
    # Modify the targets to match the new seed column names and
    # join them to the seed table
    if secondary_seed is not None:
        targets = {**primary_targets, **secondary_targets_mod}
    else:
        targets = primary_targets
    
    for name, target_df in targets.items():
        # Get the name of the geo column
        geo_colname = next((col for col in target_df.columns if col.startswith("geo_")), None)
        
        # Reshape the target table to match seed column names
        melted = pd.melt(
            target_df, 
            id_vars=[geo_colname], 
            var_name="key", 
            value_name="target"
        )
        melted["key"] = f"{name}." + melted["key"] + ".target"
        
        # Pivot back to wide format
        target_wide = melted.pivot(index=geo_colname, columns="key", values="target").reset_index()
        
        # Join to seed
        seed = pd.merge(seed, target_wide, on=geo_colname, how="left")
    
    # Calculate average, min, and max weights and join to seed
    # If there are multiple geographies in the first primary target table,
    # then min and max weights will vary by geography
    first_target_name = next(iter(primary_targets))
    first_target = primary_targets[first_target_name]
    geo_colname = next((col for col in first_target.columns if col.startswith("geo_")), None)
    
    # Count records by geography
    recs_by_geo = seed.groupby(geo_colname).size().reset_index(name="count")
    
    # Calculate total target by geography
    target_total = pd.melt(
        first_target, 
        id_vars=[geo_colname], 
        var_name="category", 
        value_name="value"
    ).groupby(geo_colname)["value"].sum().reset_index(name="total")
    
    # Calculate weight scales
    weight_scale = pd.merge(target_total, recs_by_geo, on=geo_colname)
    weight_scale["avg_weight"] = weight_scale["total"] / weight_scale["count"]
    weight_scale["min_weight"] = min_ratio * weight_scale["avg_weight"]
    weight_scale["max_weight"] = max_ratio * weight_scale["avg_weight"]
    
    # Join weight scales to seed
    seed = pd.merge(seed, weight_scale, on=geo_colname)
    
    # Initialize convergence variables
    iter_count = 1
    converged = False
    prev_weights = None
    
    # Main IPU loop
    while not converged and iter_count <= max_iterations:
        # Loop over each target and update weights
        for seed_attribute in seed_attribute_cols:
            # Create lookups for targets list
            target_tbl_name = seed_attribute.split(".")[0]
            target_name = f"{seed_attribute}.target"
            
            # Get the name of the geo column
            target_tbl = targets[target_tbl_name]
            geo_colname = next((col for col in target_tbl.columns if col.startswith("geo_")), None)
            
            # Group by geography and adjust weights
            seed_grouped = seed.groupby(geo_colname)
            
            # For each geography, adjust weights
            for geo, group in seed_grouped:
                # Calculate the total weighted attribute value
                total_weight = (group[seed_attribute] * group["weight"]).sum()
                
                # Get the target value
                target = group[target_name].iloc[0]
                
                # Calculate the adjustment factor
                if total_weight > 0:
                    factor = target / total_weight
                else:
                    factor = 1
                
                # Apply the factor to weights where attribute > 0
                mask = (seed[geo_colname] == geo) & (seed[seed_attribute] > 0)
                seed.loc[mask, "weight"] = seed.loc[mask, "weight"] * factor
                
                # Apply constraints
                # Implement the floor on zero weights
                seed.loc[mask, "weight"] = seed.loc[mask, "weight"].clip(lower=weight_floor)
                
                # Cap weights to multiples of the average weight (not applicable if target is 0)
                if target > 0:
                    min_w = seed.loc[seed[geo_colname] == geo, "min_weight"].iloc[0]
                    max_w = seed.loc[seed[geo_colname] == geo, "max_weight"].iloc[0]
                    seed.loc[mask, "weight"] = seed.loc[mask, "weight"].clip(lower=min_w, upper=max_w)
        
        # Determine percent differences (by geo field)
        saved_diff_tbl = None
        pct_diff = 0
        
        for seed_attribute in seed_attribute_cols:
            # Create lookups for targets list
            target_tbl_name = seed_attribute.split(".")[0]
            target_name = f"{seed_attribute}.target"
            target_tbl = targets[target_tbl_name]
            
            # Get the name of the geo column
            geo_colname = next((col for col in target_tbl.columns if col.startswith("geo_")), None)
            
            # Calculate differences for each geography
            diff_rows = []
            
            for geo in seed[geo_colname].unique():
                mask = (seed[geo_colname] == geo) & (seed[seed_attribute] > 0)
                if not any(mask):
                    continue
                
                # Get the target
                target = seed.loc[mask, target_name].iloc[0]
                
                # Calculate the total weighted attribute
                total_weight = (seed.loc[mask, seed_attribute] * seed.loc[mask, "weight"]).sum()
                
                # Calculate differences
                diff = total_weight - target
                abs_diff = abs(diff)
                pct_diff_val = diff / (target + 0.0000001)  # Avoid dividing by zero
                
                # Only include if absolute difference is significant
                if abs_diff > absolute_diff:
                    diff_rows.append({
                        "geo": geo,
                        "attribute": seed_attribute,
                        "target": target,
                        "total_weight": total_weight,
                        "diff": diff,
                        "abs_diff": abs_diff,
                        "pct_diff": pct_diff_val
                    })
            
            # Create a DataFrame from the collected rows
            if diff_rows:
                diff_tbl = pd.DataFrame(diff_rows)
                
                # Find the worst percent difference
                max_pct_diff = diff_tbl["pct_diff"].abs().max()
                
                if max_pct_diff > pct_diff:
                    pct_diff = max_pct_diff
                    saved_diff_tbl = diff_tbl
                    saved_category = seed_attribute
                    saved_geo = geo_colname
        
        # Test for convergence
        if iter_count > 1:
            rmse = np.sqrt(mean_squared_error(prev_weights, seed["weight"]))
            pct_rmse = rmse / np.mean(prev_weights) * 100
            converged = pct_rmse <= relative_gap
            
            if verbose:
                print(f"\rFinished iteration {iter_count}. %RMSE = {pct_rmse:.4f}", end="")
        
        prev_weights = seed["weight"].copy()
        iter_count += 1
    
    if verbose:
        print("\n" + ("IPU converged" if converged else "IPU did not converge"))
        
        if saved_diff_tbl is None:
            print(f"All targets matched within the absolute_diff of {absolute_diff}")
        else:
            print("Worst marginal stats:")
            # Find the row with the worst percent difference
            worst_row = saved_diff_tbl.loc[saved_diff_tbl["pct_diff"].abs().idxmax()]
            print(f"Category: {saved_category}")
            print(f"{saved_geo}: {worst_row['geo']}")
            print(f"Worst % Diff: {worst_row['pct_diff'] * 100:.2f}%")
            print(f"Difference: {worst_row['diff']:.2f}")
    
    # return seed
    # Set final weights into primary seed table
    # Also include average weight and distribution info
    primary_seed = primary_seed.copy()
    # primary_seed["weight"] = seed[primary_seed[primary_id].values]["weight"].values
    # primary_seed["avg_weight"] = seed[primary_seed[primary_id].values]["avg_weight"].values
    # primary_seed["weight_factor"] = primary_seed["weight"] / primary_seed["avg_weight"]
    
    primary_seed["weight"] = seed["weight"].values
    primary_seed["avg_weight"] = seed["avg_weight"].values
    primary_seed["weight_factor"] = primary_seed["weight"] / primary_seed["avg_weight"]
    
    # If the average weight is 0 (meaning the target was 0) set weight
    # and weight factor to 0
    primary_seed.loc[primary_seed["avg_weight"] == 0, "weight"] = 0
    primary_seed.loc[primary_seed["avg_weight"] == 0, "weight_factor"] = 0
    
    # Create the result dictionary
    result = {}
    result["weight_tbl"] = primary_seed
    
    # Create a histogram of weight distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(primary_seed["weight_factor"], bins=10, color="darkblue", edgecolor="gray")
    ax.set_xlabel("Weight Ratio = Weight / Average Weight")
    ax.set_ylabel("Count of Seed Records")
    ax.set_title("Weight Distribution")
    result["weight_dist"] = fig
    
    # Compare resulting weights to initial targets
    primary_comp = compare_results(primary_seed, primary_targets)
    result["primary_comp"] = primary_comp
    
    if secondary_seed is not None:
        # Add geo fields to secondary seed
        geo_cols = [col for col in primary_seed.columns if col.startswith("geo_")]
        seed_with_geo = pd.merge(
            secondary_seed,
            primary_seed[[primary_id, "weight"] + geo_cols],
            on=primary_id
        )
        
        # Run the comparison against the original, unscaled targets
        # and store in 'result'
        secondary_comp = compare_results(seed_with_geo, secondary_targets)
        result["secondary_comp"] = secondary_comp
    
    return result


def ipu_matrix(mtx, row_targets, column_targets, **kwargs):
    """
    Balance a matrix given row and column targets.
    
    This function simplifies the call to ipu() for the simple case of a matrix
    and row/column targets.
    
    Parameters
    ----------
    mtx : numpy.ndarray
        A 2D matrix to balance
    row_targets : array-like
        A vector of targets that the row sums must match
    column_targets : array-like
        A vector of targets that the column sums must match
    **kwargs
        Additional arguments that are passed to ipu(). See ipu() for details.
        
    Returns
    -------
    numpy.ndarray
        A matrix that matches row and column targets
    """
    # Convert matrix to a seed table
    rows, cols = mtx.shape
    seed = []
    
    for i in range(rows):
        for j in range(cols):
            seed.append({
                "row": i,
                "col": j,
                "weight": mtx[i, j],
                "geo_all": 1,
                "id": i * cols + j
            })
    
    seed = pd.DataFrame(seed)
    
    # Create target tables
    targets = {}
    
    # Row targets
    row_target = pd.DataFrame({"geo_all": [1]})
    for i in range(rows):
        row_target[str(i)] = [row_targets[i]]
    targets["row"] = row_target
    
    # Column targets
    col_target = pd.DataFrame({"geo_all": [1]})
    for j in range(cols):
        col_target[str(j)] = [column_targets[j]]
    targets["col"] = col_target
    
    # Run IPU
    ipu_result = ipu(seed, targets, **kwargs)
    
    # Convert result back to matrix
    final = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            final[i, j] = ipu_result["weight_tbl"].loc[ipu_result["weight_tbl"]["id"] == idx, "weight"].values[0]
    
    return final
