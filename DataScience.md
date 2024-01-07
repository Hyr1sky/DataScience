---
tags:
  - DataScience
  - ML
---
by Hyr1sky_He

> *To begin with, I think you need to know a couple of principles in Machine Learning World and Data Science World, that is we should try to vectorize all the information first, then we should observe them from the level of features, or from an abstract space, which also means embeddings and mapping is key to understanding data. Last but not least, treat problem from the aspect of distribution.*
# 1. Analysis Method

## 1.1 Feature description analysis
### 1.1.1 PCA
> Principal Component Analysis
> 
> (Also closely associated with SVD)

[用最直观的方式告诉你：什么是主成分分析PCA](https://www.bilibili.com/video/BV1E5411E71z/?share_source=copy_web&vd_source=7d2cf6f427cab8ff5afa3cb534b98123)
## 1.2 Correlation & Regression

**Correlation analysis is the foundation and prerequisite for regression analysis.**
> Only when there is a strong correlation between variables does it make sense to conduct regression analysis to seek the specific form of their correlation. 

**Regression analysis, in turn, is an in-depth and extended exploration of correlation analysis.**
>Correlation analysis relies on regression analysis to demonstrate the specific form of the quantitative relationship in phenomena.

### 1.2.1 Pearson Correlation Coefficient
Notation: $r$
$r$ is the so-called `pearson correlation coefficient`, the value range is $[-1, 1]$
It is used to assess the linear relationship between two continuous variables.
### 1.2.2 Spearman Correlation Coefficient
Notation: $\rho$ 
In statistics, Spearman’s rank correlation coefficient or Spearman’s ρ, named after Charles Spearman is a **nonparametric measure of rank correlation** (statistical dependence between the rankings of two variables). It assesses how well the relationship between two variables can be described using a **monotonic function**.
### 1.2.3 Linear Regression
**LSM** : Least square method
### Logistic Regression
Actually it is a classification problem, mapping all the variables to $[0, 1]$ that indicates the probability of a certain class.

Some activation functions should be mentioned here, such as `sigmoid`, `ReLU`, `tanh`, etc.
## 1.3 Clustering 
To begin with, we have to introduce **Distance Measure** . That is how we evaluate the quality of a cluster.
- Euclidean Distance (L2)
- Manhattan Distance (L1)
- Lr Norm
- Sup Distance (Chebyshev)
- Mahalanobis Distance
- Cosine Similarity
- Fuzzy Distance
### 1.3.1 K-Means
Still the mostly used algorithm.
### 1.3.2 Hierarchical Clustering
**Similarity** between clusters, how is it measured? Since we are dealing with points in space, distance is often used as a metric. There are generally three main approaches:

1. **Single Linkage:** Also known as nearest-neighbor, it calculates the distance between two clusters by considering the closest pair of samples, one from each cluster. This method is prone to a phenomenon called "Chaining," where clusters might be merged due to a few close points, even if the overall distance between clusters is large. This effect can lead to loosely connected clusters.

2. **Complete Linkage:** In complete linkage, the distance between two clusters is determined by the farthest pair of points, one from each cluster. This approach is the opposite extreme of single linkage, and it imposes strong restrictions. Both single and complete linkage suffer from the problem of being sensitive to specific data points without considering the overall characteristics of data within clusters.

3. **Average Linkage:** Average linkage computes the distance between two clusters by averaging the distances between all pairs of points, one from each cluster. This method tends to provide more reasonable results. However, it may be influenced by outliers, and a variation of this method involves using the median of pairwise distances.
### 1.3.3 DBSCAN
> Density-Based Spatial Clustering of Applications with Noise

Given a dataset ($D=\{x^{(1)}, x^{(2)}, ..., x^{(m)}\})$ :

1. **$\varepsilon$-Neighborhood (Eps):** For $x^{(j)} \in D$, its $\varepsilon$-neighborhood contains all samples in $D$ whose distance from $x^{(j)}$ is less than or equal to $\varepsilon$.

2. **MinPts:** The minimum number of samples within the $\varepsilon$-neighborhood.

3. **Core Object:** If the $\varepsilon$-neighborhood of $x^{(j)}$ contains at least $\text{MinPts}$ samples ($|N_{\varepsilon}(x^{(j)})| \geq \text{MinPts}$), then $x^{(j)}$ is a core object.

4. **Directly Density-Reachable:** If $x^{(j)}$ is in the $\varepsilon$-neighborhood of $x^{(i)}$ and $x^{(i)}$ is a core object, then $x^{(j)}$ is directly density-reachable from $x^{(i)}$. This relation is usually not symmetric unless $x^{(j)}$ is also a core object.

5. **Density-Reachable:** For $x^{(i)}$ and $x^{(j)}$, if there exists a sample sequence $p_1, p_2, ..., p_n$ where $p_1=x^{(i)}$, $p_n=x^{(j)}$, $p_1, p_2, ..., p_{n-1}$ are all core objects, and $p_{i+1}$ is directly density-reachable from $p_i$, then $x^{(j)}$ is density-reachable from $x^{(i)}$. This relation is transitive but not symmetric.

6. **Density-Connected:** For $x^{(i)}$ and $x^{(j)}$, if there exists $x^{(k)}$ such that both $x^{(i)}$ and $x^{(j)}$ are density-reachable from $x^{(k)}$, then $x^{(i)}$ and $x^{(j)}$ are density-connected. This relation is symmetric.

7. **Density-Based Cluster:** The largest set $C$ derived from density-reachable relations, where $C$ satisfies the following properties:
   - **Connectivity:** $x^{(i)} \in C$, $x^{(j)} \in C \rightarrow x^{(i)}$ and $x^{(j)}$ are density-connected.
   - **Maximality:** $x^{(i)} \in C$, $x^{(j)}$ is density-reachable from $x^{(i)} \rightarrow x^{(j)} \in C$.
### 1.3.4 Methods to evaluate clusters
- **Davies-Bouldin Index**
- **Dunn Index**

## 1.4 Association

### 1.4.1 Apriori Algorithm
Quite like a iterative algorithm 
[Apriori Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/apriori-algorithm/)

It introduces a concept about *frequent itemset*. 
### 1.4.2 FP-Growth Algorithm
Construct a tree-like structure to find the frequent itemset, the longer the twigs are, the more frequent it shows up in data.
[FP-growth Algorithm](https://zhuanlan.zhihu.com/p/117598874)

The FP-growth algorithm, while efficient in discovering frequent itemsets, cannot be directly applied to discover association rules. The execution speed of the FP-growth algorithm is faster than the Apriori algorithm, typically exhibiting performance improvements by more than two orders of magnitude.

The FP-growth algorithm only requires two passes over the dataset. The process of discovering frequent itemsets is as follows:
1. Construct the FP-tree.
2. Mine frequent itemsets from the FP-tree.

# 2. Machine Learning
#ML I wouldn't show much details in this part because we already cover this column in other subjects.

**From a data perspective, traditional machine learning tasks is like...**
- Classification
- Regression
- Clustering
- Collaborative Filtering
- Dimensionality Reduction   [Refer](https://www.cnblogs.com/SrtFrmGNU/p/7195867.html)

## 2.1 Classification
### 2.1.1 KNN
very naive and straight forward, "instance-based learning"
### 2.1.2 Decision Tree
Improvement: Random Forest
- bagging
- boosting
**Ensemble Algorithms**
### 2.1.3 Bayes
**classification evaluation methods**
- Accuracy 
- Precision
- Recall
For visualization, use `Precision Recall Curve`, `Reciver Operating Characteristic`.
### 2.1.4 SVM
[[cs229-notes3]]
## 2.2 Neural Network
At this part, I will just show a glimpse of NN, details can be found in ML notes and DL notes.

- Activation Function
- Loss Function
- Back Propagation (params update)
- Gradient Descent
- Evaluation Metrics
- Convolution Neutal Network
- Pooling 
- Classic CNN model
### Params and FLOPs for layer
#### 1. Convolution Layer
$params = C_o \times (k_w \times k_h \times C_i + 1)$
$FLOPs = [(C_i \times k_w \times k_h) + (C_i \times k_w \times k_h - 1) + 1] \times C_o \times W \times H$ (Former indicates the times of multiplies and the latter one indicates the plus)

#### 2. Fully Connected Layer
$params = (I + 1) \times O = I \times O + O$
$FLOPs = [I + (I - 1) + 1] \times O = (2 \times I) \times O$ (I represents the number of weights)

### Gradient Explosion & Gradient Vanishing
[LSTM如何来避免梯度弥散和梯度爆炸](https://www.zhihu.com/question/34878706)

# 3. Big Data
## 3.1 Definition
## 3.2 Hadoop
### 3.2.1 Configuration
### 3.2.2 MapReduce
## 3.3 The Big Three
> Google made it.
> 1. GFS
> 2. BIG TABLE
> 3. MapReduce


