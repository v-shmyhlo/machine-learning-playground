import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, cmap=plt.cm.Spectral):
  # Set min and max values and give it some padding
  offset = 0.5
  x_min, x_max = X[0, :].min() - offset, X[0, :].max() + offset
  y_min, y_max = X[1, :].min() - offset, X[1, :].max() + offset
  h = 0.01
  # Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  # Predict the function value for the whole grid
  Z = model(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  # Plot the contour and training examples
  plt.scatter(X[0, :], X[1, :], c=y, cmap=cmap)
  plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
