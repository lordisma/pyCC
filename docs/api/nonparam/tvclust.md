# TVClust: Clustering Bayesiano No Paramétrico con Información Relacional

TVClust (Two-View Clustering) es un modelo de clustering no paramétrico diseñado para incorporar información externa en forma de restricciones suaves (side information), como indicaciones de que ciertos pares de instancias deberían —o no deberían— pertenecer al mismo grupo. Fue propuesto como una solución robusta y escalable al problema de clustering con restricciones inciertas o ruidosas.

## Motivación

En muchas aplicaciones reales —como biología computacional, visión por computador, o análisis de redes sociales— disponemos no solo de datos numéricos sino también de información relacional entre pares de objetos:

- "Estas dos proteínas probablemente cumplen la misma función" (may-link).
- "Estos dos documentos hablan de temas diferentes" (may-not-link).

Esta información puede provenir de expertos, heurísticas o metadatos, pero suele ser incompleta o ruidosa. TVClust está diseñado para aprovechar este tipo de conocimiento sin requerir que sea preciso ni completo.

## ¿Qué es "Two-View Clustering"?

TVClust considera que los datos provienen de dos vistas independientes que comparten una misma estructura de clustering latente:

1. La vista de los datos: vectores en \(\mathbb{R}^d\).
2. La vista de las restricciones: una matriz binaria \(E\) que codifica relaciones de tipo "may-link" o "may-not-link" entre pares de instancias.

Ambas vistas son modeladas conjuntamente bajo un marco bayesiano no paramétrico.

## Fundamento Probabilístico

TVClust combina dos modelos generativos:

### 1. Mezcla de Procesos de Dirichlet (DPM)
- Modela la distribución de las instancias \(\{\mathbf{x}_i\}\) como una mezcla infinita de componentes.
- No requiere fijar el número de clusters a priori.
- Cada instancia puede iniciar un nuevo grupo con probabilidad proporcional a un parámetro \(\alpha\).

### 2. Modelo Gráfico para las Restricciones
- La matriz \(E\) de relaciones se modela como un grafo aleatorio condicional a las asignaciones de cluster.
- Si dos instancias pertenecen al mismo cluster, la probabilidad de un may-link es alta.
- Si pertenecen a clusters diferentes, la probabilidad de may-not-link es alta.

De esta manera, se busca una partición que sea coherente tanto con los datos como con las relaciones observadas.

## Inferencia Bayesiana

El modelo realiza inferencia conjunta sobre:
- Las asignaciones de cluster \(\mathbf{z}\).
- Los parámetros de cada cluster \(\theta_k\).
- Las relaciones esperadas entre instancias.

Se utiliza un procedimiento de inferencia tipo Gibbs sampling, que permite actualizar iterativamente las asignaciones condicionales a los datos y a las relaciones observadas.

## Ventajas de TVClust

✅ No requiere definir el número de clusters.  
✅ Tolera restricciones inconsistentes o ruidosas.  
✅ Ajusta automáticamente el grado de confianza en las relaciones.  
✅ Escalable y extensible a nuevos tipos de información relacional.

## Ejemplo de Aplicación

Supongamos que tenemos una colección de imágenes, y además de sus características visuales (histogramas, embeddings, etc.), disponemos de un conjunto de pares de imágenes anotadas como “parecen similares” o “parecen distintas”. TVClust usará ambas fuentes para encontrar una agrupación coherente, incluso si algunas de esas relaciones están equivocadas.

## Relación con RDP-means

TVClust es un modelo generativo probabilístico. A partir de su formulación, se puede derivar un algoritmo determinista —llamado RDP-means— aplicando un límite asintótico de varianza cero (small-variance asymptotics).

Esto permite usar TVClust como base para:

- Algoritmos deterministas eficientes (como RDP-means).
- Extensiones bayesianas más sofisticadas (por ejemplo, TVClust con kernels o datos secuenciales).

## Referencia

Khashabi, D., Wieting, J., Liu, J.Y., & Liang, F. (2015).  
"Clustering With Side Information: From a Probabilistic Model to a Deterministic Algorithm".  
Journal of Machine Learning Research (JMLR), 1–48.

## API

::: clustlib.nonparam.tvclust.TVClust