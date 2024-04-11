from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np

def plot_delaunay_points(points, delaunay):
    """
    Grafica los puntos y la triangulación de Delaunay.
    """
    plt.triplot(points[:, 0], points[:, 1], delaunay.simplices.copy())
    plt.plot(points[:, 0], points[:, 1], 'o')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triangulación de Delaunay')
    plt.grid(True)
    plt.axis('equal')
    plt.show()





