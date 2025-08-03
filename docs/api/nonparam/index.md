# Métodos No Paramétricos en Clustering

En el contexto del aprendizaje no supervisado, los métodos de clustering buscan descubrir estructuras latentes en los datos sin conocer etiquetas previas. Tradicionalmente, muchos algoritmos requieren fijar de antemano el número de clusters (como K-means). Los métodos no paramétricos eliminan esta necesidad, permitiendo que la complejidad del modelo se adapte automáticamente a los datos.

## ¿Qué significa "no paramétrico"?

Un método no paramétrico no presupone una estructura rígida (por ejemplo, un número fijo de parámetros como el número de clusters). En lugar de eso, la complejidad del modelo crece con la cantidad de datos. Esto permite capturar patrones más flexibles y adaptativos.

En clustering, esto significa:

- No se requiere fijar el número de clusters \( K \) a priori.
- El número de grupos se infiere directamente de los datos.
- Se permite modelar relaciones complejas y estructura jerárquica o superpuesta.

## Motivación

Fijar \( K \) en clustering clásico puede ser arbitrario y problemático. En cambio, los modelos no paramétricos permiten que el número de grupos crezca con la información observada, lo que los hace ideales para:

- Problemas con información incompleta o ruidosa.
- Datos dinámicos o no estacionarios.
- Escenarios con estructuras complejas o jerárquicas.

## Modelos No Paramétricos Probabilísticos

Una de las herramientas más potentes en este campo son los procesos de Dirichlet (DP), que permiten definir una distribución de probabilidad sobre particiones de los datos:

- Se parte de la idea de una mezcla infinita de distribuciones.
- Cada instancia puede generar un nuevo cluster con cierta probabilidad.
- El número de clusters crece con los datos, pero de forma controlada.

Uno de los modelos más representativos es el Dirichlet Process Mixture Model (DPMM), que generaliza los modelos de mezcla gaussianos sin requerir número de componentes fijo.

## Clustering con Información Relacional: TVClust

TVClust (Two-View Clustering) es un modelo bayesiano no paramétrico que combina:

1. Un modelo de mezcla para los datos (basado en procesos de Dirichlet).
2. Un modelo gráfico para la información relacional entre pares de instancias (restricciones suaves tipo may-link y may-not-link).

Este enfoque permite aprovechar tanto las características de los datos como conocimiento externo ruidoso, sin necesidad de especificar el número de clusters.

## De lo Probabilístico a lo Determinista: RDP-means

Al aplicar el análisis de varianza pequeña (small-variance asymptotics) sobre TVClust, se obtiene un algoritmo determinista similar a K-means, pero no paramétrico: RDP-means.

Este algoritmo:

- Genera nuevos clusters automáticamente.
- Minimiza una función objetivo que penaliza distancias internas y violaciones de restricciones.
- Se comporta como una versión robusta y flexible de K-means, especialmente útil con restricciones ruidosas.

## Ventajas de los Métodos No Paramétricos

✅ Adaptan su complejidad a los datos.  
✅ No requieren elección arbitraria de hiperparámetros como \( K \).  
✅ Son robustos frente a ruido e incertidumbre.  
✅ Pueden incorporar conocimiento externo (relacional) de forma natural.  

## Conclusión

Los métodos no paramétricos representan una poderosa alternativa al clustering tradicional, particularmente en escenarios complejos donde no es posible predefinir estructuras. TVClust y su versión determinista RDP-means son ejemplos representativos de esta clase de algoritmos, combinando modelos bayesianos, restricciones suaves y escalabilidad.

