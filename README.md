# Information Retrieval Assignment 2 UvA

This project implements TF-IDF, Word2Vec, Doc2Vec, LSI and LDI methods for ranking documents based on queries. For this process the Associated Press dataset is used. All methods can be implemented and evaluated with the code given below.

## Running the code

### Word2Vec

The main file for the Word2Vec method is word2vec.py. This file implements the Word2VecRetrieval class, which contains all necessary methods to implement Word2Vec. A object can be created by using.

```
word2vec = Word2VecRetrieval(5, 25000, 200, 1000, 'word2vec.pkl')
```

Where the first argument is the window size, the second one is the vocabulary size, the third one the embedding size, the fourth one the batch size for training the Skipgram model, and the last argument is the name of the file to which the model can be saved (or retrieved when one was already trained).

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
