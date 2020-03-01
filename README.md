# Information Retrieval Assignment 2 UvA

This project implements TF-IDF, Word2Vec, Doc2Vec, LSI and LDI methods for ranking documents based on queries. For this process the Associated Press dataset is used. All methods can be implemented and evaluated with the code given below.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Running the code

### Word2Vec

The main file for the Word2Vec method is word2vec.py. This file implements the Word2VecRetrieval class, which contains all necessary methods to implement Word2Vec. A object can be created by using.

```
word2vec = Word2VecRetrieval(5, 25000, 200, 1000, 'word2vec.pkl')
```

Where the first argument (5) is the window size, the second one (25000) is the vocabulary size, the third one (200) the embedding size, the fourth one (1000) the batch size for training the Skipgram model, and the last argument ('word2vec.pkl') is the name of the file to which the model can be saved (or retrieved when one was already saved). It will take some time to initialize the class the first time, as the dataset has to be downloaded, formatted and saved.

To use the Word2VecRetrieval, a Skipgram model should be trained. This can be done by running:


```
word2vec.train()
```

We achieved satisfactory performance after 150.000 training steps with a batch size of 1000, a window size of 5, a vocabulary size of 25.000 and an embedding size of 200. Every 20.000 training steps, the model (default: word2vec_model.pt) and the embedding (word2vec_embedding.pkl) are saved. To test intuitively whether or not the skipgram performs reasonbly, you can run:

```
word2vec_search.skipgram_search('art', n=11)
```

This will return the n most similar words to the first argument. Of course, the most similar word will always be the word itself. When results are satisfactory, the Skipgram can then be used to rank documents. To process all the documents before ranking them, you need to run:

```
word2vec.make_doc_repr()
```

This embeds all documents in the dataset, and stores the resulting vectors in the file doc_embeds.pkl. After this is done, retrieval can be performed of all documents in the AP dataset. When running word2vec.py with a trained model and embedded documents, it will automatically evaluate on all of the given queries and their relevance labels for all documents.

Manually, a document search can be done by running:

```
word2vec_search.search(doc_embeds, query_text)
```

where doc_embeds are the document embeddings previously created, and query_text a string that contains a random query. The output will be a sorted list of all document id's and their corresponding score given the query (cosine similarity).

### Latent Semantic Indexing

The LSI model is implemented in `lsi.py`. The model can be trained, saved and evaluated by calling this file from the command line. Optional parameters are `-embedding`, which can be either "tfidf" or "bow", `-num_topics`, an integer denoting the number of topics to train for and a boolean flag `--evaluate` which if set to true will execute a search for the optimal number of topics over the range [10,50,100,500,1000,2000].

For example, the following will train, save and evaluate an LSI model with 100 topics on tf-idf representations of the AP dataset:

`python lsi.py -embedding tfidf -num_topics 100`

As the LSI model is implemented as a class, it may also be imported and configured in a python script. For efficiency reasons, in the file `gensim_corpus.py` a class is defined which saves to and loads from disk BoW and Tf-idf transformations of a given dataset (so as not to have to create bow and tf-idf representations again for every subsequent model to be trained). The GensimCorpus class is handed to the LSI model for training. For example:

```
from lsi import LatentSemanticIndexing
from gensim_corpus import GensimCorpus

import read_ap

docs_by_id = read_ap.get_processed_docs()

# the embedding parameter determines the embedding created of the original docs
crp = GensimCorpus(docs_by_id, embedding="tfidf")
model = LatentSemanticIndexing(crp,  num_topics=200)
```

This model can now be used to retrieve documents based on an input query, like so:

 ```
 model.search("i like macdonalds hamburgers")
 ```


### Latent Dirichlet Allocation

The LDA model is implemented in `lda.py`. The model can be trained, saved and evaluated from the command line. Optional parameters are `-embedding` which can be either "tfidf" or "bow" and `num_topics`, determining the number of topics the LDA needs to train for. 

For example the following will train, save and evaluate an LSI model with 100 topics on tf-idf representations of the AP dataset:

`python lda.py -embedding tfidf -num_topics 100`

The LDA model is also implemented as a class and can therefore be imported and defined from within other python scripts. An example similar to the LSI one above:

```
from lsi import LatentDirichletAllocation
from gensim_corpus import GensimCorpus

import read_ap

docs_by_id = read_ap.get_processed_docs()

# the embedding parameter determines the embedding created of the original docs
crp = GensimCorpus(docs_by_id, embedding="tfidf")
model = LatentDirichletAllocation(crp,  num_topics=200)
```

This model can now be used to retrieve documents based on an input query, like so:

 ```
 model.search("i like macdonalds hamburgers")
 ```

## Built With

* [Gensim](https://radimrehurek.com/gensim/)
* [Pytorch](https://pytorch.org/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
