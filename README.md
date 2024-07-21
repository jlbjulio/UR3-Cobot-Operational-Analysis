# Proyecto Final de Analisis de datos y toma de decisiones
Proyecto Final de Analisis de datos y toma de decisiones

#Análisis de Datos de Robot UR3 CobotOps

Archivo directivo a Dashboard : http://gtfojulio.pythonanywhere.com/

Este proyecto se centra en el análisis de agrupamiento del conjunto de datos UR3 CobotOps. El conjunto de datos incluye datos de series temporales multidimensionales del cobot UR3, que ofrecen información sobre parámetros operativos y fallas para el aprendizaje automático en robótica y automatización.

El proyecto emplea técnicas de agrupamiento para analizar el conjunto de datos UR3 CobotOps. La metodología principal implica:

- Preprocesamiento de datos: manejo de valores faltantes, normalización de datos y codificación de variables categóricas.
- Agrupación: aplicación de varios algoritmos de agrupación para identificar patrones y anomalías.
- Visualización: creación de visualizaciones para interpretar los resultados de la agrupación.

Los parámetros en el conjunto de datos UR3 CobotOps están directamente relacionados con la operación y el rendimiento del cobot. 

1.	Corrientes eléctricas:
  - Indican el consumo de energía de los motores en cada articulación
  -	Pueden ayudar a detectar irregularidades o ineficiencias en el movimiento
3.	Temperaturas: 
  -	Monitorean las condiciones térmicas de motores y componentes
  -	Cruciales para prevenir el sobrecalentamiento y asegurar un rendimiento óptimo
4.	Velocidades en las articulaciones (J0-J5): 
  -	Representan el movimiento de cada una de las seis articulaciones
  -	Importantes para analizar patrones de movimiento y eficiencia
5.	Corriente del gripper: 
  -	Se relaciona con la energía utilizada por el efector final (pinza)
  -	Puede indicar la fuerza de agarre y la interacción con objetos
6.	Recuento de ciclos de operación: 
  -	Registra cuántas veces el cobot ha realizado sus tareas programadas
  -	Útil para programar mantenimiento y analizar la vida útil
7.	Paradas protectoras: 
  -	Registran cuándo se activan las características de seguridad del cobot
  -	Críticas para garantizar una operación segura alrededor de humanos
8.	Pérdidas de agarre: 
  -	Indican instancias donde el gripper falló en mantener sujeto un objeto
  -	Importantes para el control de calidad y el análisis de la tasa de éxito de las tareas


Estos parámetros proporcionan colectivamente una visión integral del estado operativo del cobot, su rendimiento y posibles problemas. Son cruciales para:
  -	Optimización del rendimiento
  -	Mantenimiento predictivo
  - onitoreo de seguridad
  - Control de calidad en procesos industriales


Temperatura J0, J1, J2, J3, J4, J5: 

- Estas son las temperaturas de cada una de las seis articulaciones del cobot.
-	J0 a J5 representan las seis articulaciones del robot, desde la base hasta el efector final.
-	Medir la temperatura de cada articulación es crucial para detectar sobrecalentamiento y prevenir daños.
  
Speed J0, J1, J2, J3, J4, J5:

-	Estas son las velocidades de rotación de cada articulación.
-	Indica qué tan rápido se está moviendo cada articulación en un momento dado.
-	Es importante para analizar la dinámica del movimiento y la eficiencia de las operaciones.
  
Current J0, J1, J2, J3, J4, J5:

-	Se refiere a la corriente eléctrica que consume cada motor de las articulaciones.
- Proporciona información sobre el esfuerzo que está realizando cada motor.
-	Útil para detectar anomalías en el consumo de energía o posibles fallos mecánicos.
  
En el cobot UR3, generalmente:

- J0: Base giratoria 
- J1: "Hombro" - primera articulación principal 
- J2: "Codo" - segunda articulación principal 
- J3: Primera articulación de la "muñeca" 
- J4: Segunda articulación de la "muñeca" - suele permitir el giro 
- J5: Tercera articulación de la "muñeca" - típicamente permite la rotación final del efector

Gráficos de clustering (K-Means, Jerárquico y DBSCAN): 
-	Eje X: Componente PCA 1
-	Eje Y: Componente PCA 2
  
Interpretación: Estos valores no tienen un "mejor" o "peor" caso. Son simplemente coordenadas en un espacio bidimensional que representan las dos características principales extraídas de los datos originales. Los puntos que están más cerca entre sí en este espacio son más similares en los datos originales.

Gráfico del método del codo: 
-	Eje X: Número de clusters
-	Eje Y: SSE (Suma de Errores Cuadráticos)
  
Interpretación: Valores más bajos de SSE son generalmente mejores, ya que indican una menor variación dentro de los clusters. Sin embargo, el objetivo es encontrar un "codo" en la curva, donde aumentar el número de clusters no produce una reducción significativa en el SSE. Este punto de inflexión sugiere el número óptimo de clusters. 

Gráfico K-distancia de DBSCAN: 
-	Eje X: Puntos de datos ordenados por distancia
-	Eje Y: Epsilon (distancia)
  
Interpretación: No hay un "mejor" valor aquí. El objetivo es identificar un "codo" en la curva, similar al método del codo. Este codo sugiere un buen valor para el parámetro epsilon en DBSCAN, que es la distancia máxima entre dos muestras para que una se considere en el vecindario de la otra. 

Gráfico de series temporales: 
-	Eje X: Índice (tiempo)
-	Eje Y: Valor de la variable (temperatura o corriente)
  
Interpretación: 
-	Para temperaturas: Generalmente, valores más bajos son mejores, ya que temperaturas más altas pueden indicar sobrecalentamiento o ineficiencia.
-	Para corrientes: La interpretación depende del contexto específico del robot. Valores consistentes y dentro del rango esperado son generalmente mejores. Picos o valores muy altos podrían indicar un problema.
