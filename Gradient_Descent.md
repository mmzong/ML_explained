# Gradient Descent

## Synonyms
* error function = cost function
* gradient = slope or rate of change

## Definition
**Gradient Descent** is a generic optimization algorithm that is used on a wide variety of problems. Here we will talk about how it is used in Machine Learning. 

The general goal of Gradient Descent is to tweak parameters iteratively to *minimize* a cost function. If you are lost on a mountain in a dense fog and you're trying to get to the bottom as quickly as you can, a good strategy would be to go downhill in the direction of the steepest local slope that you can feel with your feet. This is exactly what Gradient Descent does, the slope of the mountain is the cost function you are trying to find the minima of. For this explanation we will use *Mean Squared Error (MSE)* as our cost function. 

Here are the conceptual steps of Gradient Descent: 
1. It starts with **random initialization** where it fills $\theta$ ($\theta$ is a model parameter, it will make more sense as you read on) with random values
2. At each step, it measures the gradient of the cost function (MSE), and tries to decrease it by taking one baby step at a time in the downslope direction until the algorithm *converges* to a minimum, or finds the minimum of the cost function. 
3. The size of these steps is called the **learning rate** hyperparameter. A smaller learning rate means smaller steps. If the learning rate is too small, the algorithm will have to go through many iterations (a long time) to converge. However, if the learning rate is too high, it's possible for the algorithm to jump across a valley to the otherside of a convex-function (imagine a U-shaped function), possibly at a higher point than before, making the algorithm diverge to larger and larger values, never finding a good solution. 
    * The MSE cost function for a Linear Regression model happens to be a convex function. A **convex function** is one in which a line drawn between any 2 points on the graph lies on the graph or above it. This means that there are no local minima, only one global minimum. It is also a continuous function with a slope that never changes abruptly. 
        * The consequence is that Gradient Descent is guaranteed to approach arbitrarily close to the global minimum (if you wait long enough and if the learning rate is not too high). 

Mathematically, here is how step 2 is carried out:
* The gradient descent function computes the gradient of the cost function (MSE) with respect to each model parameter $\theta_j$.
* It does this by taking the partial derivative of the cost function (MSE), with respect to parameter $\theta_j$. This is displayed below. This formula tells you how much the cost function (MSE) changes if you change $\theta_j$ just a little.

Partial Derivative of the cost function:
```math
\frac{d}{d\theta_j}MSE(\theta_j) = \frac{2}{m} \sum_{i=1}^{m} 
    (\theta^{T} X^{(i)} - y^{(i)}) x^{(i)}_j
```

* Instead of computing each partial derivative individually (there would be one for each model parameter), you can use the **gradient vector** to compute them all in one go.
    * The gradient vector contains all partial derivatives (one for each model parameter) as the first item to the right of the $=$ sign.
    * Notice that this formula involves calculations over the full training set $X$ at each Gradient Descent step. This is why the algorithm is called **Batch Gradient Descent**, because it uses the whole batch of training data at every step
        * As a result, it is very slow on large training sets. However, Gradient Descent scales well with the number of features. Training a Linear Regression model with hundreds of thousands of features is much faster with Gradient Descent compared to the Normal Equation or SVD decomposition. 

Gradient Vector of the cost function:
```math
\nabla_{\theta}MSE(\theta) = \begin{bmatrix} 
    \frac{d}{d\theta_0}MSE(\theta) \\ 
    \frac{d}{d\theta_1}MSE(\theta) \\
    ...
    \\
    \frac{d}{d\theta_n}MSE(\theta)
    \end{bmatrix} 

    = \frac{2}{m} X^{T} (X\theta - y)
```

* The calculated Gradient Vector $\nabla_{\theta}MSE(\theta)$ tells us the direction of the steepest increase (uphill) in the cost function (MSE). Since we are looking to take a step in the opposite direction (downhill) to find the minimum of the cost function, we do the following to determine the size of the downhill step:
    1. multiply the gradient vector $\nabla_{\theta}MSE(\theta)$ by the learning rate $\eta$
    2. subtract $\nabla_{\theta}MSE(\theta)$ from $\theta$.

Gradient Descent step:
```math
\theta^{(next step)} = \theta - \eta*\nabla_{\theta}  MSE(\theta)
```

**Stochastic Gradient Descent**
*** Add sentence about it

## Implementation
*** Add different example using library of Gradient Descent

```
eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * (X_b.T.dot(theta) - y)
    theta = theta - eta * gradients

# resulting theta
theta
array([[ 4.21509616],
       [ 2.77011339]])
```

Note:
* Ensure that all features have a similar scale when using Gradient Descent, otherwise it will take much longer to converge
    * use Scikit-Learn's `StandardScaler` class.


*** Site textbook 

