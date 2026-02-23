import random
from collections import Counter
from arboles_numericos import entrena_arbol

def bootstrap(datos):
    """
    Genera un subconjunto bootstrap del mismo tamaño que datos
    usando selección aleatoria con reemplazo
    """
    n = len(datos)
    return [random.choice(datos) for _ in range(n)]


def entrena_bosque( datos, target, M, max_profundidad=None, acc_nodo=1.0, min_ejemplos=0, variables_seleccionadas=None):
    """
    Entrena un bosque aleatorio como una lista de árboles
    """
    bosque = []

    for _ in range(M):
        datos_bootstrap = bootstrap(datos)

        # Default: mayoría del subconjunto
        clases = Counter(d[target] for d in datos_bootstrap)
        clase_default = clases.most_common(1)[0][0]

        arbol = entrena_arbol(
            datos_bootstrap,
            target,
            clase_default,
            max_profundidad=max_profundidad,
            acc_nodo=acc_nodo,
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=variables_seleccionadas
        )

        bosque.append(arbol)

    return bosque


def predice_bosque(bosque, ejemplo):
    """
    Predice la clase de un ejemplo usando mayoria de votos de los árboles del bosque
    """
    predicciones = [arbol.predice(ejemplo) for arbol in bosque]
    return Counter(predicciones).most_common(1)[0][0]


def predice_bosque_datos(bosque, datos):
    """
    Predice la clase de múltiples instancias usando un bosque aleatorio, tira paro para las pruebas
    """
    return [predice_bosque(bosque, d) for d in datos]