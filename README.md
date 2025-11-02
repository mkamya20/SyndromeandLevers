# Syndrome and Levers Detection Project

A comprehensive computational framework for detecting **phenotypic syndromes** — groups of traits that appear together in species — with particular focus on identifying ecological drivers versus genetic pleiotropy through phylogenetic analysis.

## Overview

This project analyzes trait co-occurrence patterns across species to identify **syndromes**: sets of traits that cluster together phenotypically. The key innovation is distinguishing between:

- **Ecologically-driven syndromes**: Traits that co-occur in phylogenetically distant species (indicating environmental selection)
- **Pleiotropic syndromes**: Traits that co-occur in closely related species (indicating shared genetic pathways)

By integrating phenotypic clustering, local dependence analysis, and phylogenetic distance metrics, the framework identifies candidate syndromes that may result from complex ecological interactions rather than simple genetic correlations.

## Key Features

### 1. **Species Clustering with Phylogenetic Context**
- **UMAP + HDBSCAN** clustering to identify phenotypically similar species groups
- **Phylogenetic distance integration** to detect distantly related species clustering together
- **Driving trait identification** (low within-cluster, high across-cluster variance)

### 2. **Local Dependence Detection Methods**
- **k-NN Local Pearson Correlation**: Identifies localized correlation hotspots with statistical significance testing
- **Gaussian Mixture Model (GMM) Regimes**: Finds distinct correlation regimes across trait space
- **PCA-Based Analysis**: Principal Component Analysis for massive screening of trait pairs
  - Local PCA with sliding windows
  - Regime-based PCA analysis
  - Phylogenetic signal integration

### 3. **Advanced Syndrome Detection**
- **Density-based quantiles**: Divides trait space using density clustering
- **Phylogenetic PCA correlation**: Integrates phylogenetic distance with PCA analysis
- **Gaussian smoothing**: Smooths correlation profiles for better peak detection
- **Peak detection**: Identifies significant hotspots in smoothed correlation profiles
- **Harmonic mean scoring**: Combines multiple metrics ensuring all components are strong

### 4. **Statistical Rigor**
- **Permutation testing** for p-value calculation
- **Benjamini-Hochberg False Discovery Rate (FDR)** correction
- **Bootstrap confidence intervals** for correlation estimates
- **Adaptive windowing** with statistical confidence measures

### 5. **Massive Screening Capability**
- Screens **10,000+ trait pairs** efficiently
- Automated candidate syndrome identification
- CSV export of all results for further analysis

## Project Structure

```
SyndromeAndLevers/
├── data/
│   ├── master.dataset.final.v2.csv          # Main trait dataset
│   ├── Fly_Traits.0.1_Jan8_2024.csv         # Fly-specific traits
│   └── FlyTree_*.tre                         # Phylogenetic tree files
├── dev/
│   ├── Local_Dependence_Methods.ipynb       # Main analysis notebook
│   ├── Species_Clustering.ipynb              # Species clustering analysis
│   └── Trait_Pairs.ipynb                     # Trait pair analysis
├── reports/
│   └── SyndromesAnalysis_v1.pdf             # Analysis reports
└── notes/
    └── thoughts.md                           # Project notes and methodology
```

## Methodology

### Syndrome Detection Strategy

1. **Species Clustering**: Group species by phenotypic similarity using UMAP dimensionality reduction and HDBSCAN clustering
2. **Phylogenetic Analysis**: Calculate patristic (phylogenetic) distances between species
3. **Local Dependence Testing**: For each trait pair, test for localized correlation hotspots
4. **Syndrome Scoring**: Combine multiple signals:
   - High local correlation in specific trait regions
   - High phylogenetic distance (indicating ecological drivers)
   - High PC1 variance (strong local structure)
   - Peak detection in smoothed correlation profiles
5. **Candidate Identification**: Rank trait pairs by syndrome score and statistical significance

### Why This Approach?

Traditional global correlation analysis misses **localized dependencies** — trait pairs that are strongly correlated in specific regions of trait space but appear uncorrelated globally. These "islands of dependence" are hallmark patterns of syndromes.

Additionally, by integrating phylogenetic distance, we can distinguish:
- **True syndromes**: Correlated traits in phylogenetically distant species (ecological convergence)
- **Genetic correlations**: Correlated traits in closely related species (pleiotropy)

## Installation

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn scipy
pip install umap-learn hdbscan biopython
pip install statsmodels  # Optional: for FDR correction
pip install dcor  # Optional: for distance correlation
```

### Data Requirements

1. **Trait dataset**: CSV file with species as rows and traits as columns
2. **Phylogenetic tree**: Newick format tree file (.tre) with species names matching the dataset

## Usage

### Basic Workflow

1. **Load and prepare data** (`Local_Dependence_Methods.ipynb` - Cell 1):
   ```python
   # Data is automatically loaded and preprocessed:
   # - Log-transform skewed traits
   # - Z-score all traits
   # - Align species between trait data and phylogenetic tree
   ```

2. **Run species clustering** (Cells 3-7):
   ```python
   # Clusters species and identifies driving traits
   # Calculates phylogenetic distances
   ```

3. **Calculate syndrome priors** (Cell 8):
   ```python
   # Prior-based syndrome detection using phylogenetic context
   ```

4. **Run massive screening** (Cell 10 or 16):
   ```python
   # Screens thousands of trait pairs
   # Identifies candidate syndromes
   ```

5. **Visualize results** (Cell 17):
   ```python
   # Creates comprehensive visualizations:
   # - Correlation profiles with peak detection
   # - Phylogenetic distance profiles
   # - Trait scatter plots with density contours
   # - Syndrome component scores
   ```

### Example: Testing a Specific Trait Pair

```python
from dev.Local_Dependence_Methods import advanced_syndrome_detection

# Get trait data
trait_a = trait_data_zscored['trait_name_A'].values
trait_b = trait_data_zscored['trait_name_B'].values

# Run advanced syndrome detection
result = advanced_syndrome_detection(
    trait_a, trait_b, phylo_dist_matrix,
    n_regimes=3, window_size=10, smoothing_factor=1.5
)

# Check syndrome score
print(f"Syndrome Score: {result['syndrome_score']:.3f}")
print(f"Mean PC1 Variance: {result['mean_pc1_variance']:.3f}")
print(f"Detected Peaks: {result['n_peaks']}")
```

## Output Files

The analysis generates several CSV files:

- `pca_local_results.csv`: Local PCA analysis results for all tested pairs
- `pca_regime_results.csv`: Regime-based PCA analysis results
- `pca_syndrome_candidates.csv`: Ranked candidate syndromes with scores
- `knn_local_results_*.csv`: k-NN local correlation results for specific pairs
- `gmm_components_*.csv`: GMM component summaries
- `gmm_assignments_*.csv`: Per-point GMM component assignments

## Key Metrics

### Syndrome Score Components

1. **PC1 Variance**: Proportion of variance explained by first principal component (higher = stronger local structure)
2. **Mean Correlation**: Average correlation across sliding windows
3. **Mean Phylogenetic Distance**: Average phylogenetic distance in correlated regions (higher = more likely ecological driver)
4. **Peak Count**: Number of detected peaks in smoothed correlation profile

### Statistical Significance

- **p-values**: Permutation-based p-values for local correlations
- **q-values**: FDR-corrected p-values (Benjamini-Hochberg)
- **Confidence Intervals**: Bootstrap 95% CI for correlation estimates

## Visualization

The framework provides comprehensive visualizations:

1. **Correlation Profiles**: Raw and smoothed correlation across trait space with detected peaks
2. **Phylogenetic Distance Profiles**: Mean phylogenetic distance across trait windows
3. **Trait Scatter Plots**: 2D scatter with density contours
4. **Syndrome Component Scores**: Bar charts of normalized scores for each component
5. **GMM Component Visualization**: Colored scatter plots showing distinct correlation regimes

## Citation

If you use this framework in your research, please cite:

```
Syndrome and Levers Detection Project
Computational framework for phenotypic syndrome identification
with phylogenetic integration
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional detection methods
- Performance optimizations
- Visualization improvements
- Documentation enhancements

## Acknowledgments

This project implements methods for detecting phenotypic syndromes through local dependence analysis and phylogenetic integration, with applications to evolutionary ecology and trait co-evolution.

---

**Note**: This is an active research project. Methods and implementations are continuously being refined and improved.

