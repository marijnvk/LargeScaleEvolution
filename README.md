# Large Scale Evolution

This is a generalized implementing of the evolutionary optimization framework presented in [this paper by Google Brain](https://arxiv.org/abs/1703.01041). The original was intended to evolve convolutional neural network structures. This implementation can easily be adapted to any optimization problem. An example is included that naively optimizes the [Rastrigin optimization test function](https://en.wikipedia.org/wiki/Rastrigin_function), as well as a simple implementation for optimizing convolutional neural networks. If run on a POSIX-compliant system (Linux systems, for example), the implementation can circumvent the use of locks by depending on the file system for concurrent processing. On non-POSIX-compliant systems (Windows, for example), locks have to be used to prevent population corruption. Note that on Windows 10, you can use the framework in a POSIX-compliant environment through the [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux), though the environment currently does not have support for GPU access. You can use Python's *os.name* variable to determine whether a system is POSIX-compliant.

## Requirements

This project requires Python 3.6+ and no additional packages, unless you want to run the CNN example. In that case, you will also need:

* keras
* numpy
* scikit-learn
* tensorflow (or tensorflow-gpu for GPU-based training)

These packages are all available through [Anaconda/Miniconda](https://anaconda.org/).

## Files

* `lse.py`: Infrastructure for large scale evolution.
* `lse_cnn`: Mutation and evaluation functions for evolving CNNs on CIFAR-10.
* `lse_main`: Command-line interface.
* `lse_monitor`: Example of how to monitor the population while the evolutionary process is running.
* `lse_rastrigin`: Mutation and evalutation function for evolving solutions for optimizing the Rastrigin function.