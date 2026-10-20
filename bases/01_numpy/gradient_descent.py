#Minimiser (w − 3)²
#En d'autres termes, trouver la valeur de w qui rend le résulat de la fonction (w − 3)² le plus petit possible.

w = 0.0
lr = 0.1
for i in range(50):
    grad = 2 * (w - 3)      # dérivée de (w - 3)**2
    w = w - lr * grad
    print(i, round(w, 4))