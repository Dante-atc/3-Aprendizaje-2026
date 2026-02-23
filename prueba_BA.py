import utileria as ut
import bosque_aleatorio as ba
import os
import random

# Descarga y descomprime los datos

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
archivo_zip = "datos/student.zip"
archivo_csv = "datos/student-mat.csv"


# Descarga datos
if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo_csv):
    ut.descarga_datos(url, archivo_zip)
    ut.descomprime_zip(archivo_zip)

# Definir atributos
atributos = [
    'school','sex','age','address','famsize','Pstatus',
    'Medu','Fedu','Mjob','Fjob','reason','guardian',
    'traveltime','studytime','failures','schoolsup',
    'famsup','paid','activities','nursery','higher',
    'internet','romantic','famrel','freetime','goout',
    'Dalc','Walc','health','absences','G1','G2','G3'
]

# Convertir CSV de ; a , 
archivo_csv_tmp = "datos/student_mat_coma.csv"

if not os.path.exists(archivo_csv_tmp):
    with open(archivo_csv, "r") as f_in:
        with open(archivo_csv_tmp, "w") as f_out:
            for linea in f_in:
                f_out.write(linea.replace(";", ","))

#Extrae datos y convierte a numericos
datos = ut.lee_csv(
    archivo_csv_tmp,
    atributos=atributos
)

# Convertir variables que se puedan a numericas
variables_numericas = [
    'age','Medu','Fedu','traveltime','studytime',
    'failures','famrel','freetime','goout',
    'Dalc','Walc','health','absences','G1','G2','G3'
]

for d in datos:
    for v in variables_numericas:
        valor = d[v].strip().replace('"', '')
        d[v] = float(valor)

# Normalizar a 0 o 1 la variable final que es la calificacion del estudiante, 1 si aprueba, 0 si reprueba
# Ejemplo, G3 >= 10 → 1 (aprueba), < 10 → 0 (reprueba)
for d in datos:
    d["G3"] = 1 if d["G3"] >= 10 else 0

target = "G3"

# Eliminar variables que no son numericas
variables = [k for k in datos[0].keys() if k != target]
datos_limpios = []
for d in datos:
    nd = {}
    for v in variables + [target]:
        if isinstance(d[v], (int, float)):
            nd[v] = d[v]
    datos_limpios.append(nd)
    
    
# Selecciona un conjunto de entrenamiento y de validación  
random.seed(42)
random.shuffle(datos_limpios)
N = int(0.8 * len(datos_limpios))
datos_entrenamiento = datos_limpios[:N]
datos_validacion = datos_limpios[N:]


# Experimento 1: variar número de árboles
print("\nExperimento 1 Variar num de arboles (M)")
print("M\tError")
print("-" * 25)

for M in [1, 5, 10, 20, 50]:
    bosque = ba.entrena_bosque(
        datos_entrenamiento,
        target,
        M=M,
        max_profundidad=5,
        variables_seleccionadas=4
    )

    pred = ba.predice_bosque_datos(bosque, datos_validacion)
    error = sum(
        p != d[target] for p, d in zip(pred, datos_validacion)
    ) / len(datos_validacion)

    print(f"{M}\t{error:.3f}")

# Resultado
# M       Error
# -------------------------
# 1       0.089
# 5       0.127
# 10      0.114
# 20      0.101
# 50      0.101
# Resultado analizado: Al aumentar el número de árboles, 
# el error tiende a estabilizarse, mostrando la reducción de varianza que
# caracteriza a los bosques aleatorios.
 
# Experimento 2: variar profundidad máxima\
print("\nExperimento 2 Variar profundidad máxima")
print("Prof\tError")
print("-" * 25)

for prof in [2, 4, 6, 10, None]:
    bosque = ba.entrena_bosque(
        datos_entrenamiento,
        target,
        M=20,
        max_profundidad=prof,
        variables_seleccionadas=4
    )

    pred = ba.predice_bosque_datos(bosque, datos_validacion)
    error = sum(
        p != d[target] for p, d in zip(pred, datos_validacion)
    ) / len(datos_validacion)

    print(f"{prof}\t{error:.3f}")

# Resultado
# Prof    Error
# -------------------------
# 2       0.089
# 4       0.101
# 6       0.101
# 10      0.101
# None    0.114
# Resultado analizado: Profundidades excesivas incrementan el sobreajuste, 
# mientras que profundidades moderadas ofrecen un mejor balance sesgo–varianza.
    
# Experimento 3: variar variables por nodo
print("\nExperimento 3 Variar variables por nodo")
print("Vars\tError")
print("-" * 25)

for k in [1, 2, 4, 8]:
    bosque = ba.entrena_bosque(
        datos_entrenamiento,
        target,
        M=20,
        max_profundidad=5,
        variables_seleccionadas=k
    )

    pred = ba.predice_bosque_datos(bosque, datos_validacion)
    error = sum(
        p != d[target] for p, d in zip(pred, datos_validacion)
    ) / len(datos_validacion)

    print(f"{k}\t{error:.3f}")
    
# Resultado
# Vars    Error
# -------------------------
# 1       0.101
# 2       0.101
# 4       0.101
# 8       0.076
# Resultado analizado: Incrementar el número de variables consideradas 
# por nodo puede mejorar el desempeño cuando existen atributos altamente informativos.