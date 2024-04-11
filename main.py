# librerias

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path
import timeit
import time
import numpy as np

def generate_random_points(n):
    """
    Genera n puntos aleatorios en el plano.
    """
    np.random.seed(0)
    points = np.random.rand(n, 2) * 100
    return points






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # diccionario con las diferentes figuras
    points = {3: [[1, 1], [5, 1], [3, 3]],
              4: [[1, 1], [5, 1], [3, 3], [1, 5]],
              5: [[1,1],[5,1],[5,5],[1,5],[3,3]],
             6: [[1,1],[5,1],[5,5],[3,7],[1,5],[2,3]],
             7: [[1,1],[5,1],[5,5],[3,7],[1,5],[3,3],[2,3]],
             8: [[1,1],[5,1],[6,3],[5,5],[1,5],[3,4],[3,3],[2,3]],
             9: [[1,1],[5,1],[6,3],[5,5],[3,7],[1,5],[3,4],[3,3],[2,3]],
             10:[[1,1],[5,1],[6,3],[5,5],[3,7],[1,5],[3,4],[3,3],[2,3],[1,3]],
             11:[[1,1],[3,0],[5,1],[6,3],[5,5],[3,7],[1,5],[3,4],[3,3],[2,3],[1,3]],
             12:[[1,1],[3,0],[5,1],[6,3],[5,5],[3,7],[1,5],[2,4.5],[3,4],[3,3],[2,3],[1,3]],
             13:[[1,1],[3,0],[5,1],[8,3],[7,4],[5,5],[3,7],[1,5],[2,4.5],[3,4],[3,3],[2,3],[1,3]],
             14:[[1,1],[3,0],[5,1],[8,3],[7,4],[5,5],[3,7],[1.5,6],[1,5],[2,4.5],[3,4],[3,3],[2,3],[0,3]],
             15:[[1,1],[3,0],[5,1],[7,1.25],[8,3],[7,4],[5,5],[3,7],[1.5,6],[1,5],[2,4.5],[3,4],[3,3],[2,3],[0,3]],
             16:[[1,1],[3,0],[5,1],[7,1.25],[8,3],[7,5],[6,4],[5,5],[3,7],[1.5,6],[1,5],[2,4.5],[3,4],[3,3],[2,3],[0,3]],
             17:[[5,2],[11,2],[14,5],[20,6],[15,10],[13,8],[10,6],[8,8],[11,10],[12,11],[6,12],[5,11],[7,9],[5,8],[2,10],
                 [2,6],[3,1]],
             18:[[5,2],[5,1],[11,2],[14,5],[20,6],[15,10],[13,8],[10,6],[8,8],[11,10],[12,11],[6,12],[5,11],[7,9],[5,8],
                 [2,10],[2,6],[3,1]],
             19:[[5,2],[5,1],[8,1.5],[11,2],[14,5],[20,6],[15,10],[13,8],[10,6],[8,8],[11,10],[12,11],[6,12],[5,11],[7,9],
                 [5,8],[2,10],[2,6],[3,1]],
             20:[[5,2],[5,1],[8,1.5],[11,2],[14,5],[16,4],[20,6],[15,10],[13,8],[10,6],[8,8],[11,10],[12,11],[6,12],[5,11],
                 [7,9],[5,8],[2,10],[2,6],[3,1]],
             21:[[5,2],[5,1],[8,1.5],[11,2],[14,5],[16,4],[20,6],[17,11],[15,10],[13,8],[10,6],[8,8],[11,10],[12,11],[6,12],
                 [5,11],[7,9],[5,8],[2,10],[2,6],[3,1]],
             22:[[5,2],[5,1],[8,1.5],[11,2],[14,5],[16,4],[20,6],[17,11],[15,10],[13,8],[10,6],[8,8],[11,10],[12,11],[6,12],
                 [5,11],[7,9],[5,8],[3,7],[2,10],[2,6],[3,1]],
             23:[[5,2],[5,1],[8,1.5],[11,2],[14,5],[16,4],[20,6],[17,11],[15,10],[13,8],[12,5],[10,6],[8,8],[11,10],[12,11],
                 [6,12],[5,11],[7,9],[5,8],[3,7],[2,10],[2,6],[3,1]],
             24:[[1,1],[3,0],[6,1],[5,2],[7,2],[8,3],[7,6],[8,7],[7,8],[8,10],[6,11],[5,10],[6,9],[5,8],[4,9],[3,8],[3,6],[2,7],
                 [1,6],[2,5],[4,4],[2,3],[2,2],[1,2]],
             25:[[1,1],[3,0],[6,1],[5,2],[7,2],[8,3],[7,6],[8,7],[7,8],[8,10],[6,11],[5,10],[6,9],[5,8],[4,9],[3,8],[3,6],[2,7],
                 [1,6],[2,5],[4,4],[2,3],[2,2],[1,3],[0,2]],
             26:[[1,1],[3,0],[6,1],[5,2],[7,2],[8,3],[6,4],[7,6],[8,7],[7,8],[8,10],[6,11],[5,10],[6,9],[5,8],[4,9],[3,8],[3,6],
                 [2,7],[1,6],[2,5],[4,4],[2,3],[2,2],[1,3],[0,2]],
             27:[[1,1],[3,0],[6,1],[5,2],[7,2],[8,3],[6,4],[5,5],[7,6],[8,7],[7,8],[8,10],[6,11],[5,10],[6,9],[5,8],[4,9],[3,8],
                 [3,6],[2,7],[1,6],[2,5],[4,4],[2,3],[2,2],[1,3],[0,2]],
             28:[[1,1],[3,0],[6,1],[5,2],[7,2],[8,3],[6,4],[5,5],[7,6],[8,7],[7,8],[9,9],[8,10],[6,11],[5,10],[6,9],[5,8],[4,9],
                 [3,8],[3,6],[2,7],[1,6],[2,5],[4,4],[2,3],[2,2],[1,3],[0,2]],
             29:[[1,1],[3,0],[6,1],[5,2],[7,2],[8,3],[6,4],[5,5],[7,6],[8,7],[7,8],[9,9],[8,10],[7,10],[6,11],[5,10],[6,9],[5,8],
                 [4,9],[3,8],[3,6],[2,7],[1,6],[2,5],[4,4],[2,3],[2,2],[1,3],[0,2]],
             30:[[10,0],[11,1],[14,2],[18,0],[20,2],[16,4],[18,6],[24,8],[22,10],[20,8],[18,12],[16,10],[14,14],[10,16],[12,12],
                 [14,10],[10,8],[8,8],[12,6],[8,4],[7,6],[5,6],[6,8],[7,10],[6,13],[5,14],[6,15],[0,20],[2,10],[4,2]],
             31:[[10,0],[11,1],[14,2],[18,0],[20,2],[16,4],[18,6],[24,8],[22,10],[20,8],[18,12],[16,10],[14,14],[10,16],[12,12],
                 [14,10],[10,8],[8,8],[12,6],[8,4],[7,6],[5,6],[6,8],[7,10],[6,13],[5,14],[6,15],[0,20],[2,16],[3,12],[4,2]],
             32:[[10,0],[11,1],[14,2],[18,0],[20,2],[16,4],[18,6],[24,8],[22,10],[20,8],[18,12],[16,10],[14,14],[10,16],[12,12],
                 [14,10],[10,8],[8,7],[12,6],[8,4],[7,6],[5,6],[6,8],[7,10],[6,13],[5,14],[6,15],[0,20],[2,16],[3,12],[2,10],[4,2]],
             33:[[10,0],[11,1],[14,2],[18,0],[20,2],[16,4],[18,6],[24,8],[22,10],[20,8],[18,12],[16,10],[14,14],[10,16],[12,12],
                 [14,10],[10,8],[8,7],[12,6],[8,4],[7,6],[5,6],[6,8],[7,10],[6,13],[5,14],[6,15],[0,20],[2,16],[3,12],[2,10],
                 [0,5],[4,2]]
              }

    """
    tiempos = []
    for polygon_points in points:
        # Crear un camino para el polígono
        vertices = points.get(polygon_points)
        polygon_path = Path(vertices)

        t1 = timeit.default_timer()
        # Triangulación de Delaunay
        tri = Delaunay(vertices)
        stop1 = timeit.default_timer()

        # Visualizar el polígono inicial y la triangulación
        plt.figure(figsize=(20, 20))

        # Polígono
        plt.plot(*zip(*np.vstack([vertices, vertices[0]])), 'b-o', label='Polígono')

        t2 = timeit.default_timer()
        triangulacion_final = []
        # Filtrar aristas de la triangulación
        for simplex in tri.simplices:
            # Para cada arista de cada triángulo
            for i in range(3):
                p1, p2 = simplex[i], simplex[(i + 1) % 3]  # Índices de los puntos de la arista
                point1, point2 = tri.points[p1], tri.points[p2]
                edge_center = (point1 + point2) / 2

                # Verificar si el centro de la arista está dentro del polígono
                if polygon_path.contains_points([edge_center])[0]:
                    triangulacion_final.append([point1, point2])
                    # plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--')

        stop2 = timeit.default_timer()
        tiempos.append((stop2 - t2) + (stop1 - t1))

        for i in triangulacion_final:
            point1 = i[0]
            point2 = i[1]
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--')

        # Configurar la visualización
        plt.title('Polígono con Triangulación Interna Filtrada')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()
        # print(len(points))

        time.sleep(10)  # Sleep for 3 seconds

    tests = [i for i in range(3, 34)]
    f = open("tiempos.txt", "w")
    for i in range(len(points)):
        f.write(str(tests[i]) + ", " + str(tiempos[i]) + "\n")
    f.close()

    # plot x = n, y = tiempo
    plt.figure(figsize=(20, 20))
    plt.plot(tests, tiempos)
    plt.xlabel('Cantidad de vertices')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Tiempo de ejecución de divide and conquer: Triangulacion de polinomios')
    plt.show()
"""

    tiempos = []
    for i in range(40):
        # a=generate_random_points(16)
        points = generate_random_points(i)
        # Realizar la triangulación de Delaunay
        t2 = timeit.default_timer()
        delaunay = Delaunay(points)
        stop = timeit.default_timer()
        tiempos.append(stop-t2)
        time.sleep(3)

    tests = [i for i in range(len(tiempos))]
    f = open("tiempos2.txt", "w")
    for i in range(len(points)):
        f.write(str(tests[i]) + ", " + str(tiempos[i]) + "\n")
    f.close()

    # plot x = n, y = tiempo
    plt.figure(figsize=(20, 20))
    plt.plot(tests, tiempos)
    plt.xlabel('Cantidad de vertices')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Tiempo de ejecución de divide and conquer: Triangulacion de polinomios')
    plt.show()

