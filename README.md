# CE-ImageProducer
Python scripts and code to produce and transform images in a format for "Hierarchical Image Classification using Entailment Cone Embeddings"

The repository that implements this learning algorithm can be found [here](https://github.com/ankitdhall/learning_embeddings). This repository contains
specific scripts that were required to transform our chosen dataset (wine images) into a format that could be used to produce results using this method.

Performs the following steps:

1. Fetches a feed containing wines, information about wines and their unique identifiers.
2. Filter out underrepresented wines to make sure we have enough data
3. Download images from remote storage for each wine and use these images to produce enough samples by
adding noise and performing image transformations
4. Upload the images to a remote location and keep them there so work can be done simultaneously
5. Produce a JSON file in the format the is expected for the learning_embeddings algorithm using the chosen
hierarchical information.
