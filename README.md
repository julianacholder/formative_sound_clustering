# Sound Dataset Clustering Analysis

## Project Overview

This project applies unsupervised machine learning techniques to cluster an unlabeled dataset of 3,000 sound recordings. The analysis explores dimensionality reduction methods (PCA vs t-SNE) and compares clustering algorithms (K-Means vs DBSCAN) to automatically discover patterns in audio data.

## Objectives

- Extract meaningful features from sound recordings using Mel Spectrograms
- Analyze the necessity of dimensionality reduction for high-dimensional audio data
- Compare PCA and t-SNE for dimensionality reduction effectiveness
- Implement and evaluate K-Means and DBSCAN clustering algorithms
- Provide comprehensive performance analysis with quantitative metrics

## Dataset Information

- **Size**: 3,000 unlabeled WAV audio files
- **Duration**: Processed as 3-second clips for consistency
- **Features**: 16,640-dimensional Mel Spectrogram features per sample
- **Source**: Unlabeled sound recordings (various audio categories)

##  Technical Implementation

### Feature Extraction
- **Method**: Mel Spectrogram using librosa
- **Parameters**: 
  - `n_mels=128` (frequency bins)
  - `n_fft=2048` (FFT window size)
  - `hop_length=512` (step size)
  - `duration=3.0` seconds (fixed length)

### Dimensionality Reduction
- **PCA**: Linear dimensionality reduction
  - Explained variance: 65.8% (3 components)
  - Silhouette score: 0.531
- **t-SNE**: Non-linear dimensionality reduction
  - Silhouette score: 0.257
  - **Result**: PCA performed better for this dataset

### Clustering Algorithms
- **K-Means**: 
  - Optimal clusters: 10 (determined via elbow method)
  - Silhouette score: 0.322
  - Davies-Bouldin index: 1.041
- **DBSCAN**:
  - Clusters found: 117 (over-fragmentation)
  - Noise points: 138 (4.6%)
  - Silhouette score: 0.195

## 📈 Key Results

### Performance Comparison
| Algorithm | Silhouette Score | Davies-Bouldin Index | Number of Clusters | Cluster Balance |
|-----------|------------------|---------------------|-------------------|-----------------|
| K-Means   | **0.322**      | **1.041**        | 10                | Excellent       |
| DBSCAN    | 0.195            | 0.850               | 117               | Poor            |

### Dimensionality Reduction Analysis
| Method | Silhouette Score | Explained Variance | Cluster Separability |
|--------|------------------|-------------------|---------------------|
| **PCA**  | **0.531**    | 65.8%             | Superior            |
| t-SNE      | 0.257        | N/A               | Lower               |

## Key Findings

1. **PCA outperformed t-SNE** for this audio dataset, capturing essential linear relationships in mel spectrogram features
2. **K-Means significantly superior to DBSCAN**, creating 10 meaningful clusters vs 117 micro-clusters
3. **Dimensionality reduction essential** - reduced 16,640 dimensions to 3 while preserving clustering structure
4. **Uniform density distribution** in audio features favors K-Means over density-based clustering

##  Repository Structure

```
 sound-clustering-project          
│   └── Juliana_Holder_Formative_Sound_Clustering.ipynb  # Main analysis notebook
├──  results/
│   ├── visualizations/          # PCA, t-SNE, clustering plots
│   ├──  performance_metrics/     # Evaluation results
│   └──  cluster_analysis/        # Detailed clustering 
├──  README.md                   

```

##  Installation & Usage

### Prerequisites
```bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn
```

### Quick Start
```python
# 1. Clone the repository
git clone https://github.com/julianacholder/formative_sound_clustering.git
cd sound-clustering-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
jupyter notebook notebooks/Juliana_Holder_Formative_Sound_Clustering.ipynb
```

### Running Individual Components
```python
from src.feature_extraction import SoundFeatureExtractor
from src.dimensionality_reduction import DimensionalityReducer
from src.clustering_evaluation import ClusteringEvaluator

# Extract features
extractor = SoundFeatureExtractor()
features, filenames = extractor.load_dataset("data/unlabelled_sounds")

# Apply dimensionality reduction
reducer = DimensionalityReducer()
pca_data, explained_variance, _ = reducer.apply_pca(features)

# Perform clustering
evaluator = ClusteringEvaluator()
results = evaluator.compare_algorithms(pca_data)
```

##  Visualizations

The project includes comprehensive visualizations:

- **High-dimensional data challenges**: Pairplot attempts showing visualization impossibility
- **3D PCA vs t-SNE comparison**: Side-by-side dimensionality reduction results
- **Clustering optimization**: Elbow curves and silhouette analysis
- **Algorithm comparison**: Performance metrics and cluster distributions
- **3D cluster visualizations**: Interactive plots showing final clustering results

##  Applications

This clustering analysis can be applied to:
- **Music recommendation systems**: Grouping similar audio tracks
- **Audio content analysis**: Automatic categorization of sound libraries
- **Acoustic research**: Identifying patterns in environmental sounds
- **Music production**: Organizing sample libraries by acoustic similarity
- **Audio forensics**: Clustering similar audio signatures

##  Methodology

### 1. Data Preprocessing
- Load 3,000 WAV files with consistent 3-second duration
- Extract Mel Spectrogram features (128 mel bands × 130 time frames = 16,640 features)
- Standardize features for clustering analysis

### 2. Dimensionality Analysis
- Document challenges with direct visualization of 16,640-dimensional data
- Apply PCA and t-SNE with 3 components for comparative analysis
- Evaluate cluster separability using silhouette scores

### 3. Clustering Implementation
- Optimize K-Means using elbow method and silhouette analysis
- Configure DBSCAN with appropriate eps and min_samples parameters
- Compare algorithms using multiple evaluation metrics

### 4. Performance Evaluation
- Silhouette Score (cluster separation quality)
- Davies-Bouldin Index (cluster compactness)
- Inertia (within-cluster sum of squares)
- Visual cluster quality assessment

##  Academic Significance

This project demonstrates:
- **Practical application** of unsupervised learning to real audio data
- **Comparative analysis** of dimensionality reduction techniques
- **Rigorous evaluation** using multiple clustering metrics
- **Clear documentation** of methodology and results
- **Reproducible research** with well-structured code

##  Future Work

- **Expert validation**: Collaborate with audio engineers to validate discovered clusters
- **Advanced features**: Experiment with MFCC, spectral features, and deep embeddings
- **Temporal analysis**: Incorporate sequence modeling for longer audio clips
- **Semi-supervised learning**: Leverage partial labels for improved clustering
- **Real-time clustering**: Develop streaming algorithms for live audio analysis

##  Contact

**Student**: Juliana Crystal Holder    
**Email**: j.holder@alustudent.com

##  License

This project is created for academic purposes as part of coursework in machine learning technique II.

---

 **If you find this analysis helpful, please star the repository!**