## What are Vectors?

When people talk about the “vectors” that sit in a vector database, 
they mean embeddings—ordered lists of floating-point numbers that capture the essential properties of text, images, audio, or other unstructured data. 
**Each coordinate in that list is learned so that geometric closeness in the high-dimensional space (n-dimensiomal) corresponds to semantic closeness in the original domain.** 

In other words, two sentences about tacos end up near each other, while a sentence about quantum mechanics lands far away. 
Mathematically these objects are just vectors, but in practice they are dense, hundreds-to-thousands-dimensional arrays tuned for similarity search.


### How an embedding/vector is created?

Under the hood, an embedding is produced by a neural network that ends with a special embedding layer. 
During training that layer’s weights are updated—via back-propagation—until inputs that the loss function deems “similar” map to points that are close together. 
Classic examples include:
- word2vec (predict the surrounding words),
- CLIP (align images with their captions), and
- modern large-language-model heads that expose a ready-made /embeddings endpoint.
The moment you feed raw data through such a trained model, the activation of the embedding layer becomes your vector;
its dimensionality is fixed by the layer size, say 384 or 3072.
