# Implementation of a Restricted Boltzman Machine

This project aims to implement a Restricted Boltzman Machine - RBM to be used
to generate samples.

### Prerequisites

You will need the following packages in order to work with this implementation :

* Numpy
* Matplotlib for plotting the generated images
* Scipy -  io to load the data


### Content

In the project folder, you can find a Jupyter notebook which is a Step by Step
explanation of the implementation and you can find a python file containing a complete
implementation of the RBM as class.

In this case, I have used the Binary AlphaDigits Dataset that you can find in the following link:
http://www.cs.nyu.edu/~roweis/data.html.


### Example
You can easily use the RBM implementation as a class as follows :

First start by creating an RBM Instance and train the model using your data.
You will also need to adapt your dataset as to have as an input a matrix containing
at each row an example of your dataset. In our case, the 320 = 16 * 20, which are the dimension
of each image.

```python
from RBM import RBM
rbm = RBM(320, 100) # Create an RBM Architecture
rbm.fit(x, iter_gradient, epsilon, batch_size, verbose = False) # Train the model
```

After the training part, you can generate new images :

```python
rbm.generate_image(1, 1000)
```

## Important Notes

* In the following implementation, I am using the Gibbs Sampling approach, which is a Markov
Chain Monte Carlo (MCMC) algorithm in order to obtain a sequence of observation which can
approximate a specified probability distribution.

* I am open to all remarks and suggestions.
