# ProyectoFinalAnalisis
Proyecto Final de Analisis de datos y toma de decisiones

Los parámetros en el conjunto de datos UR3 CobotOps están directamente relacionados con la operación y el rendimiento del cobot. Veamos cómo se relaciona cada parámetro con el cobot:

1.	Corrientes eléctricas: 
  o	Indican el consumo de energía de los motores en cada articulación
  o	Pueden ayudar a detectar irregularidades o ineficiencias en el movimiento
2.	Temperaturas: 
  o	Monitorean las condiciones térmicas de motores y componentes
  o	Cruciales para prevenir el sobrecalentamiento y asegurar un rendimiento óptimo
3.	Velocidades en las articulaciones (J0-J5): 
  o	Representan el movimiento de cada una de las seis articulaciones
  o	Importantes para analizar patrones de movimiento y eficiencia
4.	Corriente del gripper: 
  o	Se relaciona con la energía utilizada por el efector final (pinza)
  o	Puede indicar la fuerza de agarre y la interacción con objetos
5.	Recuento de ciclos de operación: 
  o	Registra cuántas veces el cobot ha realizado sus tareas programadas
  o	Útil para programar mantenimiento y analizar la vida útil
6.	Paradas protectoras: 
  o	Registran cuándo se activan las características de seguridad del cobot
  o	Críticas para garantizar una operación segura alrededor de humanos
7.	Pérdidas de agarre: 
  o	Indican instancias donde el gripper falló en mantener sujeto un objeto
  o	Importantes para el control de calidad y el análisis de la tasa de éxito de las tareas


Estos parámetros proporcionan colectivamente una visión integral del estado operativo del cobot, su rendimiento y posibles problemas. Son cruciales para:
  •	Optimización del rendimiento
  •	Mantenimiento predictivo
  •	Monitoreo de seguridad
  •	Control de calidad en procesos industriales


  •  Temperatura J0, J1, J2, J3, J4, J5: 
      •	Estas son las temperaturas de cada una de las seis articulaciones del cobot.
      •	J0 a J5 representan las seis articulaciones del robot, desde la base hasta el efector final.
      •	Medir la temperatura de cada articulación es crucial para detectar sobrecalentamiento y prevenir daños.
  •  Speed J0, J1, J2, J3, J4, J5: 
      •	Estas son las velocidades de rotación de cada articulación.
      •	Indica qué tan rápido se está moviendo cada articulación en un momento dado.
      •	Es importante para analizar la dinámica del movimiento y la eficiencia de las operaciones.
  •  Current J0, J1, J2, J3, J4, J5: 
      •	Se refiere a la corriente eléctrica que consume cada motor de las articulaciones.
      •	Proporciona información sobre el esfuerzo que está realizando cada motor.
      •	Útil para detectar anomalías en el consumo de energía o posibles fallos mecánicos.
  En el cobot UR3, generalmente:
      •  J0: Base giratoria 
      •  J1: "Hombro" - primera articulación principal 
      •  J2: "Codo" - segunda articulación principal 
      •  J3: Primera articulación de la "muñeca" 
      •  J4: Segunda articulación de la "muñeca" - suele permitir el giro 
      •  J5: Tercera articulación de la "muñeca" - típicamente permite la rotación final del efector
