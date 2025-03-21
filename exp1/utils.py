import numpy as np
import matplotlib.pyplot as plt

def load_moon_dataset(N=1000, radius=10, width=6, d=2, r=5):
    ######################################################################
    # :param N: Number of moon scatter points
    # :param radius: Moon radius
    # :param width: Moon width
    # :param d: Y-axis offset
    # :param r: X-axis offset
    
    # :return: 
    # data (2*N*3) Moon dataset
    ######################################################################

    data = np.ones((2*N,3))
    np.random.seed(2024)
    pos = np.random.uniform(-width / 2, width / 2, size=N) + radius
    theta1 = np.random.uniform(0, np.pi, size=N)
    theta2 = np.random.uniform(np.pi, 2 * np.pi, size=N)  
    data[:,0] = np.concatenate([pos * np.cos(theta1) , pos * np.cos(theta2) + r])
    data[:,1] = np.concatenate([pos * np.sin(theta1), pos * np.sin(theta2) - d])
    data[:,2] = np.concatenate([np.ones(N), -1*np.ones(N)])
    return data
    

def plot_decision_boundary(model, X, y, dataset):
    # Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6,4))
    plt.title('Decision Boundary of {} Dataset'.format(str(dataset)))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, s=10, cmap=plt.cm.Spectral)
