# RDPMean

# RDP-means: Relational DP-means

RDP-means es una extensión determinista del algoritmo DP-means que incorpora información relacional (side information) en forma de restricciones débiles de tipo must-link y cannot-link. Es derivado del modelo probabilístico no paramétrico **TVClust**, el cual modela conjuntamente los datos y las restricciones como dos vistas generadas por una estructura latente común.

## Motivación

El clustering clásico no considera información adicional como relaciones entre pares de instancias. Sin embargo, en muchos contextos (biología, visión por computador, etc.), se dispone de información relacional incierta. RDP-means permite incorporar este conocimiento sin requerir un número fijo de clusters, y sin tratar las restricciones como verdades absolutas.

## Fundamento

RDP-means es el resultado de aplicar un análisis de varianza pequeña (small-variance asymptotics) al modelo bayesiano TVClust. Esta técnica, al llevar la varianza del modelo a cero, convierte el procedimiento de inferencia probabilística en un algoritmo determinista tipo K-means.

## Características Clave

- **No requiere fijar el número de clusters (K)**: el número de clusters emerge durante el proceso.
- **Incorpora información relacional**: usa una matriz de relaciones entre pares de instancias con etiquetas suaves (may-link y may-not-link).
- **Algoritmo eficiente**: tiene una complejidad comparable a K-means.

## Algoritmo

Sea:
- \( \mathbf{x}_i \in \mathbb{R}^d \) una instancia.
- \( \lambda \) un parámetro de penalización para crear nuevos clusters.
- \( E_{ij} \in \{1, 0, \text{NULL}\} \) representa relaciones may-link (1), may-not-link (0) o sin información.

El algoritmo sigue estos pasos:

1. Inicializar los centros de cluster con un punto aleatorio.
2. Para cada punto \( \mathbf{x}_i \):
   - Calcular el costo de asignación a cada centro \( \mu_j \) como:
     $$
     d(\mathbf{x}_i, \mu_j)^2 + \text{penalización por restricciones}
     $$
   - Si todos los costos superan \( \lambda \), se crea un nuevo cluster.
3. Recalcular los centros como la media de los puntos asignados.
4. Repetir hasta convergencia.

## Penalización por Restricciones

Se incorporan penalizaciones adicionales en la función de asignación en base a:

- Si se asigna \( \mathbf{x}_i \) y \( \mathbf{x}_j \) a clusters distintos pero \( E_{ij} = 1 \): penalización.
- Si se asignan al mismo cluster pero \( E_{ij} = 0 \): penalización.

Estas penalizaciones son modulares y no rígidas, lo cual otorga robustez al algoritmo frente a ruido en las restricciones.

## Referencia

Khashabi et al. (2015), "Clustering With Side Information: From a Probabilistic Model to a Deterministic Algorithm", *JMLR*.

## API

::: clustlib.kmean.rdpmean.RDPM