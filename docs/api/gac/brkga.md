# BRKGA: Biased Random-Key Genetic Algorithm

El algoritmo BRKGA (Algoritmo Genético con Claves Aleatorias y Tendencia), es una variante de los algoritmos genéticos diseñada para resolver problemas combinatorios mediante una representación basada en vectores de números reales llamados "random keys". Es especialmente útil cuando se requiere mantener una codificación continua que luego se decodifica a una solución entera o estructurada.

## ¿Qué son las Random Keys?

Una random key es un número real en el intervalo \([0, 1]\). En BRKGA, cada individuo se representa como un vector de claves aleatorias:

$$
\mathbf{r} = [r_1, r_2, \dots, r_n] \quad \text{con } r_i \in [0,1]
$$

Este vector es luego traducido (decodificado) a una solución factible usando una función determinista conocida como decodificador.

## Aplicación al Clustering

Para clustering con restricciones, la interpretación de las random keys puede realizarse, por ejemplo, asignando:

- Clusters según el orden de las claves (e.g., ordenarlas y asignar los primeros \(K\) valores como centroides).
- Clases o grupos según intervalos del valor de cada clave.
- Operadores personalizados para respetar restricciones ML/CL.

## Funcionamiento del BRKGA

El BRKGA modifica el esquema de los algoritmos genéticos clásicos de la siguiente manera:

### 1. Representación

- Los individuos son vectores de claves aleatorias.
- Se requiere una función de decodificación específica del problema.

### 2. Población Inicial

- Se generan \(P\) individuos con claves aleatorias en \([0,1]\).

### 3. Evaluación

- Cada individuo se decodifica a una solución del problema.
- Se evalúa su calidad mediante una función objetivo (fitness).

### 4. Selección

- Se selecciona un subconjunto de élite \(E \subset P\) con los mejores individuos.

### 5. Cruce Sesgado (Biased Crossover)

- Se generan nuevos individuos combinando claves de:
  - Un padre de la élite.
  - Un padre aleatorio (no élite).
- Con probabilidad \(p\) se toma la clave del padre élite, y con \(1 - p\) del padre no élite.

$$
r_i^\text{hijo} =
\begin{cases}
r_i^\text{élite} & \text{con probabilidad } p \\
r_i^\text{no élite} & \text{con probabilidad } 1-p
\end{cases}
$$

### 6. Mutación

- Algunos individuos nuevos se generan completamente al azar (mutantes).

### 7. Reemplazo

- La población siguiente se construye con:
  - Todos los individuos élite.
  - Individuos generados por cruce sesgado.
  - Mutantes aleatorios.

### 8. Criterio de Parada

- Número máximo de generaciones, convergencia o tiempo límite.

## BRKGA en Clustering con Restricciones

En problemas de clustering con restricciones, BRKGA se puede adaptar de la siguiente forma:

- Cada clave aleatoria define un orden o peso relativo de asignación de puntos a clusters.
- El decodificador debe respetar o penalizar las restricciones must-link y cannot-link.
- La función de fitness incluye:
  - Cohesión intra-cluster.
  - Penalización por restricciones violadas.

## Parámetros Típicos

- Tamaño de población \(P = 100\)–200.
- Porcentaje de élite: 15%–25%.
- Porcentaje de mutantes: 10%–20%.
- Probabilidad de herencia del élite: \(p \in [0.6, 0.8]\).

## Referencia

Gonçalves, J.F., Resende, M.G.C. (2011).  
"Biased Random-Key Genetic Algorithms for Combinatorial Optimization".  
Journal of Heuristics, 17(5), 487–525.

## API

::: clustlib.gac.brkga.BRKGA
