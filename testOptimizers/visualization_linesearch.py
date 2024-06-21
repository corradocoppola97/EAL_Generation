import matplotlib.pyplot as plt
import numpy as np

# Definisci i valori dei parametri
x_start = 0  # Punto dato
gamma = 0.5  # Scalari dati
alpha = 1


# Definisci le funzioni
def parabola(x):
    return x**2 -4*x + 4

def grad_parabola(x): #phi = grad(f(x))*d
    return 2*x - 4

def incremento(alpha):
    return -grad_parabola(x_start)*alpha

def direction(alpha):
    increment = incremento(alpha)
    return parabola(x_start + increment)

def retta(x, x_start, gamma, alpha):
    return parabola(x_start) + alpha*gamma*grad_parabola(x_start)*x

def retta_tan(x, x_start, gamma, alpha):
    return grad_parabola(x_start)*x + parabola(x_start)


#define various alphas
alpha_1 = 1
alpha_2 = 0.8
alpha_3 = 0.6

# Definisci i valori di x
x = np.linspace(0, 5, 1000)
x_base = np.linspace(0, 5, 1000)

# Calcola i valori delle funzioni
y1 = parabola(x)
y2 = retta(x_base, x_start,  gamma, alpha_1)
y3 = retta(x_base, x_start,  gamma, alpha_2)
y4 = retta(x_base, x_start,  gamma, alpha_3)


#true tangent
# y_tan = retta_tan(x_base, x_start,  gamma, alpha)

# Definisci i punti da evidenziare
x1_k = incremento(np.array([alpha_1]))
y1_k = direction(np.array([alpha_1]))

x2_k = incremento(np.array([alpha_2]))
y2_k = direction(np.array([alpha_2]))

x3_k = incremento(np.array([alpha_3]))
y3_k = direction(np.array([alpha_3]))

# y2_k = retta(x_k, x_start, gamma, alpha)

# Crea il grafico
plt.figure()

# Grafico della prima funzione (parabola) in rosso
plt.plot(x, y1, 'red', label='$x^2$')


# Grafico della seconda funzione (retta) in blu (alpha = 1)
plt.plot(x, y2, 'blue', label=f'$\\alpha = {alpha_1}$')

# Grafico della seconda funzione (retta) in blu (alpha = 0.5)
plt.plot(x, y3, 'orange', label=f'$\\alpha = {alpha_2}$')

# Grafico della seconda funzione (retta) in blu (alpha = 0.1)
plt.plot(x, y4, 'green', label=f'$\\alpha = {alpha_3}$')

#true tangent
# plt.plot(x, y_tan, 'g', label='tan')


plt.scatter(x1_k, y1_k, color='blue')  # Evidenzia i punti
plt.scatter(x2_k, y2_k, color='orange')  # Evidenzia i punti
plt.scatter(x3_k, y3_k, color='green')  # Evidenzia i punti

# Aggiungi le etichette degli assi e una legenda
plt.xlabel('$x_k$')
plt.ylabel('$f(x_k)$')
plt.legend()

plt.ylim(-4, 10)
# Mostra il grafico
plt.show()
