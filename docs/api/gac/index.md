# Algoritmos Genéticos: Optimización Evolutiva para Clustering con Restricciones

Los algoritmos genéticos (Genetic Algorithms, GA) son una clase de metaheurísticas inspiradas en la evolución biológica, y han demostrado ser especialmente útiles para resolver problemas de optimización complejos, como el clustering con restricciones.

## ¿Qué son los Algoritmos Genéticos?

Son algoritmos de búsqueda estocásticos que simulan el proceso de selección natural. Operan sobre una población de soluciones candidatas, que evolucionan generación tras generación aplicando operadores inspirados en la genética:

- Selección (reproducción de los más aptos),
- Cruce (combinación de soluciones),
- Mutación (variación aleatoria para explorar).

## Motivación

El problema de clustering con restricciones (must-link y cannot-link) es NP-completo. Además:

- La función objetivo puede ser no convexa, no diferenciable o discontinua.
- Hay múltiples objetivos en conflicto: cohesión de los grupos vs. satisfacción de restricciones.
- Las soluciones deben ser factibles o, al menos, penalizar las violaciones de forma controlada.

En este contexto, los algoritmos genéticos permiten:

- Explorar de forma amplia el espacio de soluciones.
- Manejar restricciones mediante funciones de penalización.
- Integrarse con búsquedas locales (algoritmos meméticos).

## Componentes de un GA

| Componente       | Descripción |
|------------------|-------------|
| Representación   | Codifica una solución (e.g., vector de asignación de clusters). |
| Función de fitness | Evalúa la calidad de la solución: compacidad + penalización por restricciones. |
| Selección        | Escoge individuos para reproducirse (ej. torneo, ruleta). |
| Cruce (crossover)| Combina dos soluciones para generar una nueva. |
| Mutación         | Introduce pequeñas alteraciones para explorar nuevas regiones. |
| Reemplazo        | Selecciona qué individuos sobreviven a la siguiente generación. |

## Aplicación al Clustering con Restricciones

En este dominio, cada individuo representa una asignación de instancias a clusters:

- Se define una función objetivo que combina:
  - Variación intra-cluster (como en K-means).
  - Penalización por violación de restricciones (must-link/cannot-link).
- Las mutaciones y cruces modifican asignaciones de puntos.
- Se usan operadores adaptados al problema (e.g., mutaciones de cluster, reparaciones).

Además, pueden incorporarse:

- Técnicas de elitismo para preservar buenas soluciones.
- Mecanismos adaptativos de penalización.
- Búsqueda local posterior (algoritmos meméticos).

## Ventajas

✅ No requiere derivadas ni supuestos sobre la función objetivo.  
✅ Robusto frente a óptimos locales.  
✅ Capaz de manejar restricciones blandas o inconsistentes.  
✅ Altamente paralelizable.  
✅ Flexible para múltiples objetivos.

## Limitaciones

⚠️ Costoso computacionalmente si no se implementa con eficiencia.  
⚠️ Requiere ajuste de parámetros (población, tasas de cruza/mutación).  
⚠️ Sin garantía de optimalidad (como toda metaheurística).

## Ejemplos de Aplicaciones

- SHADE: DE auto-adaptativo basado en historia de éxito (usado en clustering con restricciones).
- MOEA/D: Algoritmo evolutivo multiobjetivo basado en descomposición.
- PCCC: Algoritmo exacto híbrido que puede incorporar operadores genéticos para solución inicial o refinamiento.

## Conclusión

Los algoritmos genéticos representan una herramienta versátil y poderosa para abordar problemas de clustering con restricciones. Su naturaleza evolutiva les permite encontrar soluciones de alta calidad en espacios de búsqueda complejos, adaptándose bien a restricciones blandas, entornos ruidosos o no estructurados.

