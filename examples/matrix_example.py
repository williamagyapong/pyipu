"""
Matrix balancing example using the PyIPU package.

This example demonstrates how to use the IPU algorithm to balance a matrix
given row and column targets, which is a common use case in transportation modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyipu import ipu_matrix

def main():
    # Create a matrix
    mtx = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])
    
    # Define row and column targets
    row_targets = np.array([100, 200, 300])
    col_targets = np.array([200, 250, 150])
    
    print("Original matrix:")
    print(mtx)
    print("\nRow sums:", mtx.sum(axis=1))
    print("Column sums:", mtx.sum(axis=0))
    
    print("\nRow targets:", row_targets)
    print("Column targets:", col_targets)
    
    # Balance the matrix
    balanced_mtx = ipu_matrix(mtx, row_targets, col_targets, max_iterations=50, verbose=True)
    
    print("\nBalanced matrix:")
    print(balanced_mtx)
    print("\nBalanced row sums:", balanced_mtx.sum(axis=1))
    print("Balanced column sums:", balanced_mtx.sum(axis=0))
    
    # Calculate percent differences
    row_diff = (balanced_mtx.sum(axis=1) - row_targets) / row_targets * 100
    col_diff = (balanced_mtx.sum(axis=0) - col_targets) / col_targets * 100
    
    print("\nRow target percent differences:")
    for i, diff in enumerate(row_diff):
        print(f"Row {i}: {diff:.6f}%")
    
    print("\nColumn target percent differences:")
    for i, diff in enumerate(col_diff):
        print(f"Column {i}: {diff:.6f}%")
    
    # Visualize the original and balanced matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original matrix
    im1 = ax1.imshow(mtx, cmap='viridis')
    ax1.set_title('Original Matrix')
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            ax1.text(j, i, f'{mtx[i, j]:.1f}', ha='center', va='center', color='white')
    fig.colorbar(im1, ax=ax1)
    
    # Balanced matrix
    im2 = ax2.imshow(balanced_mtx, cmap='viridis')
    ax2.set_title('Balanced Matrix')
    for i in range(balanced_mtx.shape[0]):
        for j in range(balanced_mtx.shape[1]):
            ax2.text(j, i, f'{balanced_mtx[i, j]:.1f}', ha='center', va='center', color='white')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the row and column sums compared to targets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Row sums
    x = np.arange(len(row_targets))
    width = 0.35
    ax1.bar(x - width/2, mtx.sum(axis=1), width, label='Original')
    ax1.bar(x + width/2, balanced_mtx.sum(axis=1), width, label='Balanced')
    ax1.plot(x, row_targets, 'ro-', label='Target')
    ax1.set_xlabel('Row')
    ax1.set_ylabel('Sum')
    ax1.set_title('Row Sums vs. Targets')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Row {i}' for i in range(len(row_targets))])
    ax1.legend()
    
    # Column sums
    x = np.arange(len(col_targets))
    ax2.bar(x - width/2, mtx.sum(axis=0), width, label='Original')
    ax2.bar(x + width/2, balanced_mtx.sum(axis=0), width, label='Balanced')
    ax2.plot(x, col_targets, 'ro-', label='Target')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Sum')
    ax2.set_title('Column Sums vs. Targets')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Col {i}' for i in range(len(col_targets))])
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
