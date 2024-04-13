import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Suponiendo que tienes un conjunto de datos (x, y)
    x = np.array(tests[0:38], dtype=float)  # asegúrate de que x sea float para el log
    y = np.array(tiempos)


    # Modelo de la curva
    def curve_model(x, A, B):
        return A * x * np.log2(x) + B


    # Ajuste de curva
    popt, pcov = curve_fit(curve_model, x, y)

    # Coeficientes A y B
    A, B = popt
    print(f"Coeficientes del ajuste: A = {A}, B = {B}")

    # Generando puntos x para la curva ajustada
    x_fit = np.linspace(min(x), max(x), 400)
    # Prediciendo y con la curva ajustada
    y_fit = curve_model(x_fit, *popt)

    # Graficando los puntos originales
    plt.scatter(x, y, label='Datos originales', color='red')
    plt.plot(x, y, label='Curva datos originales', color='yellow')
    # Graficando la curva ajustada
    plt.plot(x_fit, y_fit, label='Curva ajustada', color='blue')

    # Calculando R^2
    y_pred = curve_model(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Coeficiente de correlación R^2: {r_squared}")

    plt.title('Ajuste de curva y Datos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Suponiendo que tienes un conjunto de datos (x, y)
    x = np.array(tests[0:38], dtype=float)  # asegúrate de que x sea float para el log
    y = np.array(tiempo2)


    # Ajuste de curva
    popt, pcov = curve_fit(curve_model, x, y)

    # Coeficientes A y B
    A, B = popt
    print(f"Coeficientes del ajuste: A = {A}, B = {B}")

    # Generando puntos x para la curva ajustada
    x_fit = np.linspace(min(x), max(x), 400)
    # Prediciendo y con la curva ajustada
    y_fit = curve_model(x_fit, *popt)

    # Graficando los puntos originales
    plt.scatter(x, y, label='Datos originales', color='red')
    plt.plot(x, y, label='Curva datos originales', color='yellow')
    # Graficando la curva ajustada
    plt.plot(x_fit, y_fit, label='Curva ajustada', color='blue')

    # Calculando R^2
    y_pred = curve_model(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Coeficiente de correlación R^2: {r_squared}")

    plt.title('Ajuste de curva y Datos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()






