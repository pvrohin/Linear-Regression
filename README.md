# CS541 Deep Learning - Homework 1: Linear Regression

This repository contains the implementation of Homework 1 for CS541 Deep Learning course, focusing on linear algebra operations, linear regression, and probability distributions.

## Overview

This homework assignment includes three main components:

1. **Matrix Operations (Problems 1a-1n)**: Implementation of various linear algebra operations including matrix addition, multiplication, eigenvalue decomposition, and statistical operations.

2. **Linear Regression**: Age regression implementation using analytical solution with training and testing error calculation.

3. **Probability Distributions**: Analysis of Poisson distribution data with comparison to alternative rate parameters.

## Files

- `hw1_cs541_dl.py` - Complete implementation with all solutions
- `homework1_template.py` - Template file with function signatures
- `PoissonX.npy` - Poisson distribution data for analysis
- `homework1.pdf` - Assignment instructions
- `HW1___CS541.pdf` - Course assignment PDF

## Prerequisites

### Required Python Packages

```bash
pip install numpy scipy matplotlib
```

### Required Data Files

The code expects the following data files to be present in the same directory:
- `age_regression_Xtr.npy` - Training features (48x48 images reshaped to 2304-dimensional vectors)
- `age_regression_ytr.npy` - Training labels (ages)
- `age_regression_Xte.npy` - Test features (48x48 images reshaped to 2304-dimensional vectors)  
- `age_regression_yte.npy` - Test labels (ages)

**Note**: The current implementation has hardcoded paths pointing to `/home/rohin/Downloads/` for these files. You'll need to either:
1. Download the data files and place them in the correct directory, or
2. Modify the file paths in the `train_age_regressor()` function to point to your data location.

## Usage

### Running the Complete Implementation

```bash
python hw1_cs541_dl.py
```

This will execute all three parts of the homework:
1. Matrix operations demonstrations
2. Linear regression training and testing
3. Poisson distribution analysis with visualizations

### Running Individual Components

You can also import and use individual functions:

```python
import numpy as np
from hw1_cs541_dl import problem_1a, linear_regression, train_age_regressor

# Example matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = problem_1a(A, B)

# Example linear regression
# (requires data files to be present)
train_error, test_error = train_age_regressor()
```

## Implementation Details

### Matrix Operations (Problems 1a-1n)

The implementation includes:
- Matrix addition (`problem_1a`)
- Matrix multiplication and subtraction (`problem_1b`)
- Element-wise multiplication and transpose (`problem_1c`)
- Inner product calculation (`problem_1d`)
- Matrix inverse operations (`problem_1e`, `problem_1f`)
- Row sum operations (`problem_1g`)
- Conditional mean calculation (`problem_1h`)
- Eigenvalue decomposition (`problem_1i`)
- Multivariate Gaussian sampling (`problem_1j`)
- Row permutation (`problem_1k`)
- Z-score normalization (`problem_1l`)
- Matrix replication (`problem_1m`)
- L2 distance calculation (`problem_1n`)

### Linear Regression

The linear regression implementation uses the analytical solution:
- **Training**: `w = (X^T X)^(-1) X^T y`
- **Loss Function**: Mean Squared Error (MSE)
- **Data**: 48×48 grayscale images reshaped to 2304-dimensional feature vectors

### Probability Distributions

The Poisson distribution analysis includes:
- Loading and visualization of `PoissonX.npy` data
- Comparison with alternative rate parameters (2.5, 3.1, 3.3, 3.7)
- Histogram overlays for visual comparison

## Expected Output

When running the complete script, you should see:
1. Outputs from all matrix operations
2. Training and testing errors for age regression
3. Multiple histogram plots comparing Poisson distributions
4. Interactive prompts to continue between visualizations

## Notes

- The code includes interactive prompts (`input("Press Enter to continue...")`) to pause between visualizations
- Some functions may require specific data files that are not included in this repository
- The implementation follows the exact specifications from the homework assignment
- All visualizations use matplotlib and will display in separate windows

## Colab Version

A Google Colab version is available at:
https://colab.research.google.com/drive/1TFz5FjcRDnlaV0DlD698rklgmUoKVPR1#scrollTo=FAiiHXas2gox