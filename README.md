# Building a Neural Network from Scratch

## About

*WORK IN PROGRESS*

This repo is intended as a learning resource for anyone who is new to deep learning and wants to quickly grasp the basic concepts. It contains a set of notebooks that explain and implement a set of networks of varying complexity. Each network is implemented using only vanilla Python and Numpy in an effort to understand how each step of the process works.

Ideally, you should a have a basic knowledge of python in order to understand the code fully, but I've tried to keep it relatively simple. I've also included a list of learning resources that I used to put this project together if you want to do any further reading.

## Requirements

To install the requirements run the following command:

`pip install -r requirements.txt`

The Tensorflow package is included in the requirements.txt file to easily download some of the data we want to train our models on (MNIST), and also to provide a benchmark for the performance of our models. `ipykernel` is package that will allow you to run a Jupyter Notebook in your IDE - I used VS Code for this project.

## Resources

Resources used to develop this project include:

- These blog posts written by Bala Priya C regarding binary/categorical cross entropy and the softmax function:
`https://www.pinecone.io/learn/cross-entropy-loss/`
`https://www.pinecone.io/learn/softmax-activation/`

- An explanation of the backpropagation algorithm written by Matt Mazur:
`https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/`

- A blog post discussing the implementation of a basic MLP using C++, written by Lyndon Duong:
`https://www.lyndonduong.com/linalg-cpp/`