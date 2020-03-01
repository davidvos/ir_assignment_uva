# Information Retrieval Assignment 2 UvA

This project implements TF-IDF, Word2Vec, Doc2Vec, LSI and LDI methods for ranking documents based on queries. For this process the Associated Press dataset is used. All methods can be implemented and evaluated with the code given below.

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

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
