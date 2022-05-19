# IA-2022_1

# **ÍNDICE**

- [Regresión](#Regresión)
- [Clasificación](#Clasificación)
- [Clustering](#Clustering)

# **Regresión**

## **Regresión lineal univariable**
- Intento de modelar relación entre variables ajustando una ecuación lineal a la data observada.

### **Hypothesis**

$$h(x_i) = x_i*w + b$$

A machine learning [`hypothesis`](https://machinelearningmastery.com/what-is-a-hypothesis-in-machine-learning/) is a candidate model that approximates a target function for mapping inputs to outputs. 

### **Loss function**

**MSE / Quadratic Loss / L2 Loss**

$$L(x_i) = Error = \frac{1}{2m}\sum_{i=0}^m (y_i - h(x_i)) ²$$

### **Gradient descent**

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.

$$db = \frac{1}{m}\sum_{i=0}^m(y_i - h(x_i))(-1)$$

$$dw = \frac{1}{m}\sum_{i=0}^m(y_i - h(x_i))(-x_i)$$


**Change parameter**

$$w_i = w_i - \alpha * \frac{\partial{\text{ Loss}}}{\partial{w_i}}$$

## **Regresión lineal multivariable**
- Generalización de la regresión lineal para el caso donde existe más de una variable independiente.

### **Hypothesis**
For $n$ features:

$$h(x^{(i)}) = w_0*x^{(i)}_0 + w_1 * x^{(i)}_1 + w_2 * x^{(i)}_2 + w_n * x^{(i)}_n$$

Where:

$$x_0 = \text{implicit } 1$$

And exists $n + 1$ variables to predict. Then:

Viewing at it as arrays where each row of matrix $x$ is a training example and exists $j$ training examples, we have:

$$
x = 
\begin{bmatrix} 
1 & x_{1,1} & ... & x_{1,n} \\
1 & x_{2,1} & ... & x_{2,n} \\
1 & ... & ... & ... \\
1 & x_{j,1} & ... & x_{j,n}
\end{bmatrix}_{j*(n+1)}
$$ 

And $x^{(j)}_{n}$ represents "the j-th value of the n-th feature" or "the n-th feature of the j-th training example". In addition:

$$w = \begin{bmatrix} w_0 & w_1 & ... & w_n \end{bmatrix}_{1*(n+1)}$$

$$
y = \begin{bmatrix} y_1 & y_2 & ... & y_j \end{bmatrix}_{1*j}
$$

Thus:

$$h(x_j) = x_j * w^t$$

$$
h(x_j) = 
\begin{bmatrix} 1 & x_{j,1} & ... & x_{j,n}\end{bmatrix}_{1*(n+1)} *
\begin{bmatrix} w_0 \\ w_1 \\ ... \\ w_n \end{bmatrix}_{(n+1)*1} 
= \text{model prediction}
$$




### **Loss function**

- Mean Squared Loss(MSE)
- Mean Absolute Loss(MAE)
- Huber Loss(MSE + MAE)

### **Gradient descent**

$$
\frac{\partial{L}}{\partial{w_0}} = \frac{1}{m}\sum_{i=0}^m(y_i - h(x^i))(-1) \\
\frac{\partial{L}}{\partial{w_{j\neq 0}}} = \frac{1}{m}\sum_{i=0}^m(y_i - h(x^i))(-x^{i}_{j})
$$

## **Regresión no lineal**
- Regresión en el que la data observada se modela mediante combinación no lineal de los parámetros del modelo.

## **Regresión logística**
- Método de clasificación que modela la probabilidad de un resultado discreto dado un input.
- Modela generalmente resultados binarios (T/F)

# **Clasificación**

## **Support vector machine**
- Método usado en regresión y clasificación.
- Algoritmo que trata de encontrar un hiperplano divisorio bajo ciertos límites de decisión.

## **KNN**
- Método supervisado de clasificación, el cual usa la proximidad para realizar clasificaciones o predicciones sobres la agrupación de un punto de datos individual
- Puede ser usado también para regresión
- Lazy learning algorithm

**Voting**
Plurality vote (clasificación)
Majority vote

**Formalización**



## **Decision Trees**
- Modelo de predicción

# **Clustering**

## **K-Means**
- Método de clustering no supervisado que busca dividir la data en K grupos donde cada observación pertenece al grupo con valor medio más cercano

## **Gaussian mixture model**

# **Articles**

- [Hypothesis in ML](https://machinelearningmastery.com/what-is-a-hypothesis-in-machine-learning/)
- [Common Loss functions](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)
    - MSE, MAE, MBE, Hinge Loss, Cross Entropy Loss
- [MAE MSE Hubber Loss](https://datamonje.com/regression-loss-functions/)
- [Gradient Descent](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
    - Learning rate, Cost function
- [Understanding learning rate](https://towardsdatascience.com/https-medium-com-dashingaditya-rakhecha-understanding-learning-rate-dd5da26bb6de)


# **References**

[SVM & PCA Tutorial from beginner](https://www.kaggle.com/code/faressayah/support-vector-machine-pca-tutorial-for-beginner)

[ML from scratch](https://github.com/marvinlanhenke/DataScience/tree/main/MachineLearningFromScratch)

[Multiclass SVM](https://gist.github.com/mblondel/97cffbea574a5890f0d7)

[ML-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/algorithms)