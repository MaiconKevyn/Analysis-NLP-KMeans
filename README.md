## Analysis and Validation of Resumes with Natural Language Process and Clustering 
## 

This project utilizes Natural Language Processing (NLP) and Clusterig for an effective analysis of resumes. The central method is TF-IDF and KMeans, which identifies the most relevant words in the resumes. 
Following this, clustering is done using the K-Means algorithm, grouping the resumes based on their textual characteristics.

A key step is the analysis of the centroids of each cluster, where words with high TF-IDF values are extracted. 
This is crucial for understanding the predominant skills and qualifications in each group.

The project employs Cosine Similarity to calculate the similarity between the keywords of the centroids and the resumes. 
This approach is fundamental in determining the most appropriate category for each centroid, based on the similarity of content.

 <h1 align="center"> <img src="https://github.com/MaiconKevyn/Analysis-NLP-KMeans/assets/101146083/c622f81d-4915-4703-9d35-980f42b5c563" width="450" height="350"> <img src="https://github.com/MaiconKevyn/Analysis-NLP-KMeans/assets/101146083/24fccc1f-a9ce-4195-8964-1fe8a008828b" width="350" height="350"> </h1>
<h1 align="center"> <img src="https://github.com/MaiconKevyn/Analysis-NLP-KMeans/assets/101146083/094317e0-f586-4e02-89c2-1ecae06e0c93" width="350" height="350"> </h1>

Using the confusion matrix, we evaluated the efficiency of the model that combines KMeans and TF-IDF, achieving an accuracy of 81%. 
This analysis indicates a good correlation between the model's predictions and the actual data.

The model shows good results, but there is room for improvement, given the limitations of KMeans. To increase the robustness of the model, it is suggested to explore other clustering techniques, in addition to KMeans.

To further enhance the current model utilizing KMeans and TF-IDF, several advanced techniques can be implemented:

* Experimenting with Different Clustering Methods: Beyond KMeans, algorithms such as DBSCAN, Agglomerative Clustering, or Spectral Clustering could be explored. These methods might handle clusters of varying sizes and shapes better, offering different insights into the data structure.

* Using Dimensionality Reduction: Techniques like PCA (Principal Component Analysis) or t-SNE could be applied to reduce the data's dimensionality before clustering. This might improve the performance of algorithms on complex, high-dimensional datasets.

* Incorporating Text Embedding Models: Models like Word2Vec, BERT, or GloVe could be used to create more sophisticated embeddings of resumes. These models better capture the context and semantic relationships of words, enriching the analysis.

* Topic Analysis: Methods such as LDA (Latent Dirichlet Allocation) or NMF (Non-negative Matrix Factorization) could be employed for a more detailed view of specialization areas and skills in the resumes.

* Cross-Evaluation with Other Metrics: Using other evaluation metrics like the Davies-Bouldin Index or the Calinski-Harabasz Index could offer an additional perspective on the suitability of the number of clusters and the quality of clustering.

* Fine-Tuning and Hyperparameter Optimization: Fine-tuning the hyperparameters of the clustering and vectorization models could significantly improve the results.
