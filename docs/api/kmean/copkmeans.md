# COP-KMeans

## Descripción General

**COP-KMeans** (Constrained K-Means) es un algoritmo clásico de clustering con restricciones propuesto por Wagstaff et al. (2001). Se basa en el algoritmo K-Means tradicional, pero incorpora información adicional en forma de restricciones de tipo **must-link (ML)** y **cannot-link (CL)** para guiar el proceso de particionado de los datos.

Las restricciones ML indican que dos instancias deben estar en el mismo clúster, mientras que las CL indican que dos instancias no deben estar en el mismo clúster. Estas restricciones se consideran **duras** (hard constraints), es decir, el algoritmo solamente considera válidas aquellas asignaciones que las satisfacen por completo.

## Funcionamiento del Algoritmo

El algoritmo sigue la estructura general del K-Means clásico, con la diferencia de que incorpora un paso de validación de restricciones en la fase de asignación. Los pasos básicos son:

1. **Inicialización**: Selección aleatoria de `k` centroides iniciales.
2. **Asignación de instancias**:
   - Cada instancia se asigna al centroide más cercano **si y solo si** no viola ninguna restricción (ML o CL) con las instancias previamente asignadas.
   - Si no existe un clúster válido que cumpla con las restricciones, el algoritmo **falla** y termina sin una solución.
3. **Recomputación de centroides**:
   - Para cada clúster válido, se actualiza su centroide como la media de sus instancias asignadas.
4. **Repetición**:
   - Se repiten los pasos de asignación y actualización de centroides hasta la convergencia (sin cambios en las asignaciones) o un número máximo de iteraciones.

## Formalización

- Dado un conjunto de datos: $$ X = {x_1, x_2, \dots, x_n} \subset \mathbb{R}^d $$
- Un número de clústers: $$ k $$
- Dos conjuntos de restricciones: $$ ML \subset X \times X $$ $$ CL \subset X \times X $$

El objetivo es encontrar una partición \( C = \{C_1, \dots, C_k\} \) tal que:

- Se minimice la suma de errores cuadráticos intra-clúster: \( \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2 \) 
  donde \( \mu_i \) es el centroide de \( C_i \),
- respetando las restricciones en \( ML \cup CL \).

## API

::: clustlib.kmean.copkmeans.COPKMeans

