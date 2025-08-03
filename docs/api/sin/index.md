# Introducción a la Búsqueda Local

La Búsqueda Local (Local Search) es una familia de algoritmos de optimización que exploran el espacio de soluciones vecinas a una solución actual, con el objetivo de encontrar un óptimo (local o global) para una función objetivo dada.

## Motivación

En muchos problemas de optimización, especialmente aquellos que son NP-duros como el clustering con restricciones, encontrar soluciones óptimas exactas es ineficiente o inviable. La búsqueda local ofrece una alternativa basada en mejoras progresivas, con bajo costo computacional.

## Conceptos Clave

### Solución Actual
Una posible solución del problema. En clustering, suele ser una asignación de instancias a clusters.

### Vecindario (Neighborhood)
El conjunto de soluciones que pueden obtenerse aplicando una operación elemental sobre la solución actual (p. ej., mover un punto de un cluster a otro).

### Movimiento (Move)
Una modificación concreta de la solución actual que genera una solución vecina.

### Función Objetivo
Evalúa la calidad de cada solución. En clustering con restricciones, suele combinar:
- Compacidad interna (ej. suma de distancias intra-cluster)
- Penalización por restricciones violadas (must-link / cannot-link)

## Funcionamiento General

1. Se parte de una solución inicial.
2. Se explora el vecindario inmediato buscando mejoras.
3. Se acepta el mejor vecino (o alguno aleatoriamente, según estrategia).
4. Se repite hasta cumplir un criterio de parada (estancamiento o límite de iteraciones).

## Limitaciones

- Puede estancarse en óptimos locales.
- No garantiza encontrar el óptimo global.
- Necesita un buen diseño del vecindario y la función objetivo.

## Mejora: Búsqueda Local Iterada (ILS)

ILS consiste en aplicar múltiples rondas de búsqueda local, intercaladas con perturbaciones que modifican parcialmente la solución para escapar de óptimos locales.

## Rol en Clustering con Restricciones

En algoritmos como DILS o MOEA/D, la búsqueda local permite ajustar asignaciones de puntos para mejorar la calidad del clustering sin violar (o minimizando violaciones de) las restricciones.

## Beneficios

- Simplicidad de implementación.
- Adaptabilidad a diferentes problemas.
- Eficiencia en la mejora de soluciones existentes.

