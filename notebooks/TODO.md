## Lessons so far:
- using higher-dimensional sentence embeddings (like all-mpnet-base-v2 (768 dimensions) instead of all-MiniLM-L6-v2 (384 dimensional)) does not bring a signifiant performance boost despite being much slower
- to filter some of the "Stop-Sentences" using "Robust Covariance" algorithms (like EllipticEnvelope that elimates X% of Noise) instead of simple Covariance helps a lot
- It is not obvious that using a more complex "cluster fitting" algorithm (like using a mixture of Gaussians instead of just one) enhances performance: the main reason being is that if an "embedding point" sits in between two relevant clusters of embedding, then its meaning is also in-between the meaning of the two clusters.

## TODOs
- Manually generate more complicated validation and test sets: F1 scores are too high because sets are too easy for the model. We need to provide more ambiguous/hard sentences, and create realistic conversations
- Identify more **Stop Sentences** kinds. This allows us to add new cleanup steps to pre-processing (already done are short sentences removal, links, etc,..)
- More manual analysis of misclassified sentences
- Push experiments to Streamlit Cloud to easier review and compare:
  - F1 scores on validation and test sets
  - Visualize various embeddings in lower dimensional space for ID sentences (each FAQ)
  - display cut-off histograms
  - display tables of misclassified sentences
