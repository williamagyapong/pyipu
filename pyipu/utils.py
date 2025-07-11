"""Utility functions for the IPU algorithm."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


def check_tables(primary_seed, primary_targets, secondary_seed=None, 
                secondary_targets=None, primary_id="id"):
    """
    Check seed and target tables for completeness.
    
    Given seed and targets, checks to make sure that at least one
    observation of each marginal category exists in the seed table.
    Otherwise, IPU would produce wrong answers without throwing errors.
    
    Parameters
    ----------
    primary_seed : pandas.DataFrame
        Primary seed table (e.g., household data)
    primary_targets : dict
        Dictionary of DataFrames with target marginals for primary seed
    secondary_seed : pandas.DataFrame, optional
        Secondary seed table (e.g., person data)
    secondary_targets : dict, optional
        Dictionary of DataFrames with target marginals for secondary seed
    primary_id : str, default="id"
        Column name that links primary and secondary seed tables
        
    Returns
    -------
    tuple
        (primary_seed, primary_targets, secondary_seed, secondary_targets)
    
    Raises
    ------
    ValueError
        If any validation check fails
    """
    # If person data is provided, both seed and targets must be
    if bool(secondary_seed is not None) != bool(secondary_targets is not None):
        raise ValueError("You provided either secondary_seed or secondary_targets, but not both.")
    
    ## Primary checks ##
    
    # Check that there are no NA values in seed or targets
    if primary_seed.isna().any().any():
        raise ValueError("primary_seed table contains NAs")
    
    for name, target in primary_targets.items():
        if target.isna().any().any():
            raise ValueError(f"primary_targets table '{name}' contains NAs")
    
    # Ensure that a weight field exists in the primary table.
    if "weight" not in primary_seed.columns:
        primary_seed = primary_seed.copy()
        primary_seed["weight"] = 1
    else: 
        # warn user about the side effect of the weight column
        print("Warning: The primary seed data contains a 'weight' column. This column will be treated as observation weights and updated during the IPU process. Please rename the weight column before calling ipu() if it has a different meaning!")
    
    # Check the primary_id
    secondary_seed_exists = secondary_seed is not None
    id_field_exists = primary_id in primary_seed.columns
    
    if not id_field_exists:
        if secondary_seed_exists:
            raise ValueError(f"The primary seed table does not have field, '{primary_id}'.")
        else:
            primary_seed = primary_seed.copy()
            primary_seed[primary_id] = range(1, len(primary_seed) + 1)
    
    unique_ids = primary_seed[primary_id].unique()
    if len(unique_ids) != len(primary_seed):
        raise ValueError(f"The primary seed's {primary_id} field has duplicate values.")
    
    # Check primary target tables for correctness
    for name, tbl in primary_targets.items():
        result = check_geo_fields(primary_seed, tbl, name)
        primary_seed, primary_targets[name] = result
        tbl = primary_targets[name]
        
        # Get the name of the geo field
        geo_colname = next((col for col in tbl.columns if col.startswith("geo_")), None)
        
        # Check that every non-zero target has at least one observation in the seed table
        check_missing_categories(primary_seed, tbl, name, geo_colname)
    
    ## Secondary checks (if provided) ##
    
    if secondary_seed_exists:
        # Check for NAs
        if secondary_seed.isna().any().any():
            raise ValueError("secondary_seed table contains NAs")
        
        for name, target in secondary_targets.items():
            if target.isna().any().any():
                raise ValueError(f"secondary_targets table '{name}' contains NAs")
        
        # Check that secondary seed table has a primary_id field
        if primary_id not in secondary_seed.columns:
            raise ValueError(f"The secondary seed table does not have field '{primary_id}'.")
        
        # Check that the secondary seed table does not have any geo columns
        if any(col.startswith("geo_") for col in secondary_seed.columns):
            raise ValueError("Do not include geo fields in the secondary_seed table (primary_seed only).")
        
        # Check the secondary target tables for correctness
        for name, tbl in secondary_targets.items():
            result = check_geo_fields(secondary_seed, tbl, name)
            secondary_seed, secondary_targets[name] = result
            tbl = secondary_targets[name]
            
            # Get the name of the geo field
            geo_colname = next((col for col in tbl.columns if col.startswith("geo_")), None)
            
            # Add the geo field from the primary_seed before checking
            temp_seed = pd.merge(
                secondary_seed,
                primary_seed[[primary_id, geo_colname]],
                on=primary_id
            )
            
            # Check that every non-zero target has at least one observation in the seed table
            check_missing_categories(temp_seed, tbl, name, geo_colname)
            
            # Remove geo_all from secondary seed if it was added
            if "geo_all" in secondary_seed.columns:
                secondary_seed = secondary_seed.drop(columns=["geo_all"])
                if "geo_all" not in primary_seed.columns:
                    primary_seed = primary_seed.copy()
                    primary_seed["geo_all"] = 1
    
    # Return seeds and targets in case of modifications
    return (primary_seed, primary_targets, secondary_seed, secondary_targets)


def check_missing_categories(seed, target, target_name, geo_colname):
    """
    Check for missing categories in seed.
    
    Parameters
    ----------
    seed : pandas.DataFrame
        Seed table to check
    target : pandas.DataFrame
        A single target table
    target_name : str
        The name of the target (e.g., size)
    geo_colname : str
        The name of the geo column in both the seed and target (e.g., geo_taz)
        
    Raises
    ------
    ValueError
        If a category with a non-zero target is missing from the seed table
    """
    for geo in seed[geo_colname].unique():
        # Get column names for the current geo that have a >0 target
        geo_target = target[target[geo_colname] == geo]
        non_zero_cols = [col for col in geo_target.columns 
                         if col != geo_colname and geo_target[col].sum() > 0]
        
        # Convert column names to appropriate types (numeric if possible)
        try:
            col_values = [pd.to_numeric(col) for col in non_zero_cols]
        except ValueError:
            col_values = non_zero_cols
        
        # Check if all target categories exist in the seed
        seed_values = seed.loc[seed[geo_colname] == geo, target_name].unique()
        missing = [val for val in col_values if val not in seed_values]
        
        if missing:
            raise ValueError(
                f"Marginal {target_name} category {', '.join(map(str, missing))} "
                f"missing from {geo_colname} {geo} in the seed table with a target greater than zero."
            )


def check_geo_fields(seed, target, target_name):
    """
    Check geo fields in seed and target tables.
    
    Makes sure that geographies in a seed and target table line up properly.
    
    Parameters
    ----------
    seed : pandas.DataFrame
        Seed table to check
    target : pandas.DataFrame
        A single target table
    target_name : str
        The name of the target (e.g., size)
        
    Returns
    -------
    tuple
        (seed, target) - possibly modified
        
    Raises
    ------
    ValueError
        If geo fields are not properly configured
    """
    # Require a geo field if >1 row
    # We cast col names to str to avoid issues with potential int columns
    geo_cols = [col for col in target.columns if str(col).startswith("geo_")]
    
    if len(target) > 1:
        if not geo_cols:
            raise ValueError(
                f"target table '{target_name}' has >1 row but does not have a "
                "geo column (must start with 'geo_')"
            )
    else:
        # If the table has 1 row and no geo field, add one
        if not geo_cols:
            target = target.copy()
            target["geo_all"] = 1
            seed = seed.copy()
            seed["geo_all"] = 1
    
    if len(geo_cols) > 1:
        raise ValueError(
            f"target table '{target_name}' has more than one geo column (starts with 'geo_')"
        )
    
    return (seed, target)


def scale_targets(targets, verbose=False):
    """
    Scale targets to ensure consistency.
    
    Often, different marginals may disagree on the total number of units. In the
    context of household survey expansion, for example, one marginal might say
    there are 100k households while another says there are 101k. This function
    solves the problem by scaling all target tables to match the first target
    table provided.
    
    Parameters
    ----------
    targets : dict
        Dictionary of DataFrames with target marginals
    verbose : bool, default=False
        Show a warning for each target scaled?
        
    Returns
    -------
    dict
        Dictionary with the scaled targets
    """
    targets = {name: df.copy() for name, df in targets.items()}
    global_total = None
    warning_msg = "Scaling target tables: "
    show_warning = False
    
    for i, (name, target) in enumerate(targets.items()):
        # Get the name of the geo field
        # We cast col names to str to avoid issues with potential int columns
        geo_colname = next((col for col in target.columns if str(col).startswith("geo_")), None)
        
        # Calculate total of table
        melted = pd.melt(target, id_vars=[geo_colname], var_name="category", value_name="count")
        total = melted["count"].sum()
        
        # If first iteration, set total to the global total. Otherwise, scale table
        if i == 0:
            global_total = total
        else:
            fac = global_total / total
            # Write out warning
            if fac != 1 and verbose:
                show_warning = True
                warning_msg += f" {name}"
            
            # Scale the target
            for col in target.columns:
                if col != geo_colname:
                    target[col] = target[col] * fac
    
    if show_warning:
        print(warning_msg)
    
    return targets


def balance_secondary_targets(primary_targets, primary_seed, secondary_targets, 
                             secondary_seed, secondary_importance, primary_id):
    """
    Balance secondary targets to primary.
    
    The average weight per record needed to satisfy targets is computed for both
    primary and secondary targets. Often, these can be very different, which leads
    to poor performance. The algorithm must use extremely large or small weights
    to match the competing goals. The secondary targets are scaled so that they
    are consistent with the primary targets on this measurement.
    
    If multiple geographies are present in the secondary_target table, then
    balancing is done for each geography separately.
    
    Parameters
    ----------
    primary_targets : dict
        Dictionary of DataFrames with target marginals for primary seed
    primary_seed : pandas.DataFrame
        Primary seed table (e.g., household data)
    secondary_targets : dict
        Dictionary of DataFrames with target marginals for secondary seed
    secondary_seed : pandas.DataFrame
        Secondary seed table (e.g., person data)
    secondary_importance : float
        A value between 0 and 1 signifying the importance of the secondary targets
    primary_id : str
        Column name that links primary and secondary seed tables
        
    Returns
    -------
    dict
        Dictionary of the balanced secondary targets
    """
    secondary_targets = {name: df.copy() for name, df in secondary_targets.items()}
    
    # Extract the first table from the primary target list and geo name
    pri_target_name = next(iter(primary_targets))
    pri_target = primary_targets[pri_target_name]
    pri_geo_colname = next((col for col in pri_target.columns if col.startswith("geo_")), None)
    
    for name, sec_target in secondary_targets.items():
        # Get geography field
        sec_geo_colname = next((col for col in sec_target.columns if col.startswith("geo_")), None)
        
        # If the geographies used aren't the same, convert the primary table
        if pri_geo_colname != sec_geo_colname:
            # Get a mapping from primary geo to secondary geo
            geo_mapping = primary_seed[[pri_geo_colname, sec_geo_colname]].drop_duplicates()
            geo_mapping = geo_mapping.groupby(pri_geo_colname).first().reset_index()
            
            # Join to primary target
            pri_target_with_sec_geo = pd.merge(
                pri_target,
                geo_mapping,
                on=pri_geo_colname
            ).drop(columns=[pri_geo_colname])
        else:
            pri_target_with_sec_geo = pri_target
        
        # Summarize the primary and secondary targets by geography
        pri_target_sum = pd.melt(
            pri_target_with_sec_geo, 
            id_vars=[sec_geo_colname], 
            var_name="cat", 
            value_name="count"
        ).groupby(sec_geo_colname)["count"].sum().reset_index(name="total")
        
        sec_target_sum = pd.melt(
            sec_target, 
            id_vars=[sec_geo_colname], 
            var_name="cat", 
            value_name="count"
        ).groupby(sec_geo_colname)["count"].sum().reset_index(name="total")
        
        # Get primary and secondary record counts
        pri_rec_count = primary_seed.groupby(sec_geo_colname).size().reset_index(name="recs")
        
        # Join secondary seed with primary geo info
        sec_with_geo = pd.merge(
            secondary_seed,
            primary_seed[[primary_id, sec_geo_colname]],
            on=primary_id
        )
        sec_rec_count = sec_with_geo.groupby(sec_geo_colname).size().reset_index(name="recs")
        
        # Calculate average weights
        pri_rec_count = pd.merge(pri_rec_count, pri_target_sum, on=sec_geo_colname)
        pri_rec_count["avg_weight"] = pri_rec_count["total"] / pri_rec_count["recs"]
        
        sec_rec_count = pd.merge(sec_rec_count, sec_target_sum, on=sec_geo_colname)
        sec_rec_count["avg_weight"] = sec_rec_count["total"] / sec_rec_count["recs"]
        
        # Calculate the factor
        weight_ratio = pd.merge(
            pri_rec_count[[sec_geo_colname, "avg_weight"]],
            sec_rec_count[[sec_geo_colname, "avg_weight"]],
            on=sec_geo_colname,
            suffixes=("_pri", "_sec")
        )
        weight_ratio["ratio"] = weight_ratio["avg_weight_pri"] / weight_ratio["avg_weight_sec"]
        
        # Adjust the factor based on importance
        weight_ratio["factor"] = adjust_factor(weight_ratio["ratio"], 1 - secondary_importance)
        
        # Update the secondary targets by the factor
        sec_target_with_factor = pd.merge(
            sec_target,
            weight_ratio[[sec_geo_colname, "factor"]],
            on=sec_geo_colname
        )
        
        # Apply the factor to all non-geo columns
        for col in sec_target_with_factor.columns:
            if col != sec_geo_colname and col != "factor":
                sec_target_with_factor[col] = sec_target_with_factor[col] * sec_target_with_factor["factor"]
        
        # Remove the factor column
        secondary_targets[name] = sec_target_with_factor.drop(columns=["factor"])
    
    return secondary_targets


def adjust_factor(factor, importance):
    """
    Apply an importance weight to an IPU factor.
    
    At lower values of importance, the factor is moved closer to 1.
    
    Parameters
    ----------
    factor : float or array-like
        A correction factor that is calculated using target/current
    importance : float
        A value between 0 and 1 signifying the importance of the factor.
        An importance of 1 does not modify the factor. An importance of
        0.5 would shrink the factor closer to 1.0 by 50 percent.
        
    Returns
    -------
    float or array-like
        The adjusted factor
        
    Raises
    ------
    ValueError
        If importance is not between 0 and 1
    """
    # Return the same factor if importance = 1
    if importance == 1:
        return factor
    
    if importance > 1 or importance < 0:
        raise ValueError("`importance` argument must be between 0 and 1")
    
    # Otherwise, return the adjusted factor
    adjusted = 1 - ((1 - factor) * (importance + 0.0001))
    return adjusted


def compare_results(seed, targets):
    """
    Compare results to targets.
    
    Parameters
    ----------
    seed : pandas.DataFrame
        Seed table with a weight column
    targets : dict
        Dictionary of DataFrames with target marginals
        
    Returns
    -------
    pandas.DataFrame
        DataFrame comparing balanced results to targets
    """
    comparison_tbl = []
    
    for name, target in targets.items():
        # Get the name of the geo field
        geo_colname = next((col for col in target.columns if col.startswith("geo_")), None)
        
        # Gather the current target table into long form
        target_long = pd.melt(
            target,
            id_vars=[geo_colname],
            var_name="category",
            value_name="target"
        )
        target_long["geo"] = geo_colname + "_" + target_long[geo_colname].astype(str)
        target_long["category"] = name + "_" + target_long["category"].astype(str)
        target_long = target_long.drop(columns=[geo_colname])
        
        # Summarize the seed table
        result = seed[[geo_colname, name, "weight"]].copy()
        result["geo"] = geo_colname + "_" + result[geo_colname].astype(str)
        result["category"] = name + "_" + result[name].astype(str)
        result = result.groupby(["geo", "category"])["weight"].sum().reset_index(name="result")
        
        # Join them together
        joined_tbl = pd.merge(target_long, result, on=["geo", "category"], how="left")
        
        # Append it to the master target df
        comparison_tbl.append(joined_tbl)
    
    # Combine all comparison tables
    comparison_tbl = pd.concat(comparison_tbl, ignore_index=True)
    
    # Calculate difference and percent difference
    comparison_tbl["diff"] = comparison_tbl["result"] - comparison_tbl["target"]
    comparison_tbl["pct_diff"] = round(comparison_tbl["diff"] / comparison_tbl["target"] * 100, 2)
    comparison_tbl["diff"] = round(comparison_tbl["diff"], 2)
    
    # Sort by geo and category
    comparison_tbl = comparison_tbl.sort_values(["geo", "category"])
    
    # If the temporary geo field geo_all was created, clean it up
    comparison_tbl["geography"] = comparison_tbl["geo"].str.replace(r"geo_all.*", "geo_all", regex=True)
    comparison_tbl = comparison_tbl.drop(columns=["geo"])
    
    return comparison_tbl


def process_seed_table(df, primary_id, marginal_columns):
    """
    Process a seed table for IPU.
    
    Helper function that strips columns from seed table except for the
    primary id and marginal column (as reflected in the targets tables). Also
    identifies factor columns with one level and processes them before
    one-hot encoding is applied.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The seed table to process
    primary_id : str
        Column name of the primary ID
    marginal_columns : list
        List of column names in the seed table that have matching targets
        
    Returns
    -------
    pandas.DataFrame
        Processed seed table with one-hot encoded marginal columns
    """
    # Select only the relevant columns and drop geo columns
    cols_to_keep = [col for col in df.columns if col in marginal_columns or col == primary_id]
    df = df[cols_to_keep].copy()
    
    # Handle any factors with only 1 level
    for name in marginal_columns:
        if len(df[name].unique()) == 1:
            # Get the single value
            value = df[name].iloc[0]
            
            # Create a new column with the value in the name
            new_name = f"{name}.{value}"
            df[new_name] = 1
            
            # Drop the original column
            df = df.drop(columns=[name])
            
            # Update marginal_columns
            marginal_columns = [col if col != name else new_name for col in marginal_columns]
    
    # One-hot encode the remaining marginal columns
    categorical_cols = [col for col in marginal_columns if col in df.columns]
    
    if categorical_cols:
        # Create a OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit and transform the categorical columns
        encoded = encoder.fit_transform(df[categorical_cols])
        
        # Get the feature names
        feature_names = []
        for i, col in enumerate(categorical_cols):
            for j, category in enumerate(encoder.categories_[i]):
                feature_names.append(f"{col}.{category}")
        
        # Create a DataFrame with the encoded features
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        
        # Combine with the original DataFrame (excluding the categorical columns)
        result = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    else:
        # If no categorical columns remain (all were single-value), just return the DataFrame
        result = df
    
    return result
