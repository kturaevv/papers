```python
### Reducing the Dimensionality of Data with Neural Networks

This paper introduces Autoencoders, the way to "encode" high dimensional data in lower dimensional space and using this "compressed" data
to reconstruct the original image. Basically precursors to modern generative models. The main problem was the inability of "plain" networks 
to work from the get go. To address this authors used "pretrained" network regularization with RBM's.

Paper highlights:
 - AE significantly outperformed PCA on dimensionality reduction tasks
 - First demonstration that deep autoencoders can learn meaningful compressed representations of complex data
 - Hinted on the importance of depth (deeper autoencoders yielded better results with same parameter count)
 - Introduced RBM-based pretraining as a solution to the network weight normalization problem
 - Showed that proper weight initialization through pretraining prevents networks from getting stuck in poor local optima
 - Proved that pretraining dramatically improves both final results and training convergence speed
 - Demonstrated diminishing returns of network depth - adding layers helps but with diminishing impact
 - Successfully applied AEs to document retrieval, outperforming LSA/PCA-based semantic search
```
