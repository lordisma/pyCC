# SHADE: Success-History Based Adaptive Differential Evolution

SHADE es una variante auto-adaptativa del algoritmo de evolución diferencial (Differential Evolution, DE), diseñada para resolver problemas de optimización continua con alta eficiencia y robustez. En el contexto de clustering con restricciones, SHADE ha demostrado ser eficaz para encontrar particiones de calidad optimizando una función objetivo que combina compacidad y violación de restricciones.

## ¿Qué es la Evolución Diferencial (DE)?

DE es un algoritmo evolutivo basado en poblaciones que optimiza una función objetivo real mediante la combinación lineal de soluciones. Sus operadores clave son:

- Mutación: combina soluciones para crear una nueva (vector mutante).
- Cruce (crossover): mezcla el vector mutante con uno original (vector trial).
- Selección: elige entre el vector original y el trial según el fitness.

## Motivación de SHADE

Aunque DE es potente, su rendimiento depende fuertemente de los parámetros de control:

- Factor de escala \( F \)
- Tasa de cruce \( CR \)

SHADE resuelve este problema mediante una adaptación basada en la historia de éxito, es decir, aprende qué combinaciones de parámetros han funcionado bien en el pasado y los reutiliza con probabilidad creciente.

El algoritmo presenta los siguientes fases:

1. Inicializar una población de soluciones con vectores reales.
2. Para cada generación:
   - Elegir un conjunto de padres.
   - Generar un vector mutante usando una estrategia como current-to-pbest/1.
   - Cruzar el mutante con el padre para obtener el trial.
   - Evaluar la calidad del trial.
   - Reemplazar si mejora la solución.
   - Registrar los valores de \( F \) y \( CR \) que generaron soluciones exitosas.
3. Actualizar una memoria histórica con los \( F \) y \( CR \) exitosos.
4. Repetir hasta convergencia.

## Success-History Adaptation

SHADE mantiene dos vectores de memoria:

- \( M_F \): historial de factores de escala exitosos.
- \( M_{CR} \): historial de tasas de cruce exitosas.

Para generar los parámetros de un nuevo individuo, SHADE:

- Selecciona aleatoriamente un índice \( k \) del historial.
- Usa una distribución Cauchy centrada en \( M_F[k] \) para \( F \).
- Usa una distribución Normal centrada en \( M_{CR}[k] \) para \( CR \).
- Los nuevos valores se ajustan si salen de sus rangos válidos.

Este mecanismo favorece gradualmente los valores que han tenido éxito, manteniendo diversidad.

## Aplicación a Clustering con Restricciones

En clustering con restricciones, cada individuo representa una partición del conjunto de datos:

- Puede codificarse como un vector real que luego se decodifica a una asignación discreta.
- La función de fitness incluye:
  - Variación intra-cluster (e.g., suma de distancias).
  - Penalización por violación de must-link y cannot-link.
- La estrategia current-to-pbest/1 favorece la convergencia sin perder diversidad.
- SHADE puede integrar búsquedas locales para acelerar la explotación.

## SHADE en Clustering con Restricciones

SHADE ha sido adaptado con éxito para abordar el problema de clustering con restricciones instancia-instancia, donde el objetivo es encontrar una partición de los datos que:

1. Minimice la varianza intra-cluster (cohesión).
2. Respete un conjunto de restricciones suaves tipo:
   - Must-Link (ML): dos instancias deben ir al mismo cluster.
   - Cannot-Link (CL): dos instancias no deben ir al mismo cluster.

Dado que estas restricciones pueden ser contradictorias o ruidosas, SHADE incorpora una función objetivo penalizada que permite tolerar violaciones a cambio de obtener una mejor calidad general de partición.

### Representación de la Solución

En este contexto, cada individuo en la población de SHADE representa una partición de los datos. Hay varias formas de codificar esto:

- Una codificación entera directa (no diferenciable) donde cada posición representa la asignación de un punto a un cluster.
- Una codificación con claves aleatorias (random keys) que se traduce a clusters mediante un decodificador determinista.
- Una codificación real continua donde se utiliza un proceso de agrupamiento posterior (como agrupamiento por proximidad en un espacio latente).

### Función Objetivo

La función de evaluación para cada individuo incluye:

$$
\text{Fitness}(\mathbf{l}) = \underbrace{\sum_{i=1}^K \sum_{x_j \in C_i} \|x_j - \mu_i\|^2}_{\text{Varianza intra-cluster}} + \lambda \cdot \text{Infeasibility}(\mathbf{l})
$$

Donde:

- \( \mu_i \) es el centroide del cluster \( C_i \).
- La función Infeasibility cuenta las restricciones ML y CL violadas.
- \( \lambda \) es un parámetro de penalización ajustable.

Este enfoque permite tratar restricciones como soft-constraints y controlar el grado de compromiso entre calidad de agrupamiento y cumplimiento estructural.

## Referencia 

Tanabe, R., & Fukunaga, A. (2013).  
Success-History Based Parameter Adaptation for Differential Evolution.  
IEEE Congress on Evolutionary Computation (CEC), pp. 71–78.

González-Almagro et al. (2020): SHADE se utilizó como núcleo en un algoritmo de clustering con restricciones basado en DE.

## API

::: clustlib.gac.shade.ShadeCC

