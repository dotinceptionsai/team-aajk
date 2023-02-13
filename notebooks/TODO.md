Lessons so far:
- using higher-dimensional sentence embeddings (like all-mpnet-base-v2 (768 dimensions) instead of all-MiniLM-L6-v2 (384 dimensional)) does not bring a signifiant performance boost despite being much slower
- to filter some of the "Stop-Sentences" using "Robust Covariance" algorithms (like EllipticEnvelope that elimates X% of Noise) instead of simple Covariance helps a lot
- It is not obvious that using a more complex "cluster fitting" algorithm (like using a mixture of Gaussians instead of just one) enhances performance: the main reason being is that if an "embedding point" sits in between two relevant clusters of embedding, then its meaning is also in-between the meaning of the two clusters.

- TODOs
- Manually generate more complicated validation and test sets: F1 scores are too high because sets are too easy for the model. We need to provide more ambiguous sentences, and create realistic conversations
- Identify more "Stop Sentences" to add cleanup steps to pre-processing (already done are cleanup of short sentences, links, etc,..)
- Manual analysis of misclassified sentences
- Push experiments to Streamlit to quick visualize results and filter them by embeddings model, date, score, etc,..
