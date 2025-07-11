"""
Example of using the synthesize function to create a synthetic population.

This example demonstrates how to use the IPU algorithm to generate weights
and then use the synthesize function to create a synthetic population.

_____
For this example, taken from https://datascience.oneoffcoder.com/ipf.html, we simulate a data set that has 3 fields.

- race: {white, other}
- gender: {male, female}
- weight (the weight of a person, or how heavy a person is)

Weights for each individual are sampled as follows:

- $N_{wm}(150, 1.5)$ for white $w$ male $m$
- $N_{om}(160, 1.3)$ for other $o$ male $m$
- $N_{wf}(125, 1.1)$ for white $w$ female $f$
- $N_{of}(130, 1.1)$ for other $o$ female $f$

The goal is to upsample the simulated data so that the sampled data has the following population proportions:

- race: white (58%), other (42%)
- gender: male (49%), female (51%)

We will convert these proportions to counts by applying a sample size of 10,000.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyipu import ipu, synthesize

def main():

    # Simulate data
    wm = np.random.normal(150, 1.5, 100)
    om = np.random.normal(160, 1.3, 80)
    wf = np.random.normal(125, 1.1, 200)
    of = np.random.normal(130, 1.1, 150)

    data = [{'race': 'white', 'gender': 'male', 'weight_p': w} for w in wm]
    data += [{'race': 'other', 'gender': 'male', 'weight_p': w} for w in om]
    data += [{'race': 'white', 'gender': 'female', 'weight_p': w} for w in wf]
    data += [{'race': 'other', 'gender': 'female', 'weight_p': w} for w in of]

    df = pd.DataFrame(data)
    print(df.shape)
    print(df.head())

    # check sample proportions: These don't match the population targets
    print(df['race'].value_counts(normalize=True))
    print(df['gender'].value_counts(normalize=True))

    #--- Now, let's use IPU(a.k.a IPF) to align the data to the target population
    # Specify targets (note target margins must be in frequencies)
    n = 1000
    u = pd.DataFrame({"white": 0.58*n, 'other': 0.42*n}, index=[0])
    v = pd.DataFrame({"male": 0.49*n, "female": 0.51*n}, index=[0])
    target_margins = {"race": u, "gender": v}
    print(target_margins)
    # run the IPU algorithm on the data
    result = ipu(df, target_margins)

    # Create the synthetic population
    df_syn = synthesize(result['weight_tbl'])
    print(df_syn.shape)

    # check the proportions
    print("Verifying proportions...")
    print(df_syn['race'].value_counts(normalize=True))
    print(df_syn['gender'].value_counts(normalize=True))

    # Compute the average weight by race and gender
    print("Computing the average weight by race and gender...")
    print(df_syn.groupby(['race', 'gender'])['weight_p'].mean())



if __name__ == "__main__":
    main()
