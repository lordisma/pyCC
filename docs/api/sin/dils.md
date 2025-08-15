# DILS: Dual Iterative Local Search para Clustering con Restricciones

DILS (Dual Iterative Local Search) es un algoritmo metaheurístico diseñado específicamente para abordar el problema de clustering con restricciones de tipo instancia-instancia (must-link y cannot-link). Se basa en un esquema de búsqueda local iterada que busca mantener un equilibrio entre exploración global y explotación local del espacio de soluciones.

## Motivación

El problema de clustering con restricciones es NP-completo incluso para instancias simples. Las soluciones exactas no escalan, y los métodos clásicos pueden quedar atrapados en óptimos locales. DILS fue propuesto como una alternativa robusta que combina técnicas de búsqueda local con mecanismos de escape de óptimos locales.

## Características Principales

- Capaz de manejar grandes volúmenes de restricciones (miles o millones).
- Emplea un esquema dual: mantiene dos soluciones para mejorar la diversidad.
- Balance entre exploración (diversificación) y explotación (intensificación).
- Usa una función de penalización para restricciones violadas (soft-constraints).

## Estructura del Algoritmo

DILS se basa en la metaheurística de Búsqueda Local Iterada (ILS), mejorándola con elementos duales y adaptativos:

### 1. Representación

- Cada solución es un vector de asignaciones de clusters:  
  $$
  \mathbf{l} = [l_1, l_2, ..., l_n] \quad \text{con } l_i \in \{1, ..., K\}
  $$

### 2. Función Objetivo

DILS minimiza una función compuesta por dos términos:
$$
f(\mathbf{l}) = \text{Varianza intra-cluster} + \alpha \cdot \text{Infeasibility}
$$
Donde:
- La varianza mide la cohesión interna.
- La infeasibility penaliza violaciones a restricciones must-link y cannot-link.

### 3. Búsqueda Local

Se aplica una búsqueda local sobre la solución actual, explorando vecinos generados por pequeños cambios (por ejemplo, reasignación de puntos a otros clusters).

### 4. Perturbación

Para escapar de óptimos locales, se perturba la solución parcialmente (p. ej., se reasignan aleatoriamente algunos puntos), iniciando una nueva búsqueda local.

### 5. Dualidad

DILS mantiene dos soluciones en paralelo. En cada iteración se perturba y mejora una de ellas, y se explora el espacio entre ambas mediante recombinación y aceptación adaptativa.

### 6. Criterio de Parada

Número máximo de iteraciones sin mejora o tiempo límite.

## Ventajas de DILS

- Altamente escalable y robusto ante ruido.
- No requiere número fijo de restricciones.
- Admite restricciones inconsistentes o redundantes.

## Referencia

González-Almagro et al. (2020).  
"DILS: Constrained clustering through dual iterative local search", *Computers and Operations Research*, 121, 104979.

## API

::: clustlib.sin.dils.DILS