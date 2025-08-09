# UR3 CobotOps Data Analysis

This project focuses on clustering analysis of the UR3 CobotOps dataset. The dataset includes multidimensional time series data from the UR3 cobot, providing insights into operational parameters and failures for machine learning in robotics and automation.

The project employs clustering techniques to analyze the UR3 CobotOps dataset. The main methodology includes:

- **Data Preprocessing**: Handling missing values, data normalization, and encoding categorical variables.
- **Clustering**: Applying several clustering algorithms to identify patterns and anomalies.
- **Visualization**: Creating visualizations to interpret clustering results.

The parameters in the UR3 CobotOps dataset are directly related to the operation and performance of the cobot.

1. **Electric Currents**:
   - Indicate energy consumption of the motors in each joint.
   - Can help detect irregularities or inefficiencies in movement.

2. **Temperatures**:
   - Monitor thermal conditions of motors and components.
   - Crucial for preventing overheating and ensuring optimal performance.

3. **Joint Speeds (J0-J5)**:
   - Represent the movement of each of the six joints.
   - Important for analyzing motion patterns and efficiency.

4. **Gripper Current**:
   - Related to the energy used by the end effector (gripper).
   - Can indicate grip strength and interaction with objects.

5. **Operation Cycle Count**:
   - Records how many times the cobot has performed its programmed tasks.
   - Useful for scheduling maintenance and analyzing lifespan.

6. **Protective Stops**:
   - Record when the cobot's safety features are triggered.
   - Critical for ensuring safe operation around humans.

7. **Grip Losses**:
   - Indicate instances when the gripper failed to hold an object.
   - Important for quality control and analyzing task success rate.

These parameters collectively provide a comprehensive view of the cobot's operational status, performance, and potential issues. They are crucial for:
- Performance optimization
- Predictive maintenance
- Safety monitoring
- Quality control in industrial processes

---

### Temperatures J0, J1, J2, J3, J4, J5:

- These are the temperatures of each of the cobot’s six joints.
- J0 to J5 represent the six joints of the robot, from base to end effector.
- Monitoring joint temperature is essential to detect overheating and prevent damage.

### Speed J0, J1, J2, J3, J4, J5:

- These are the rotation speeds of each joint.
- Indicate how fast each joint is moving at a given moment.
- Important for analyzing motion dynamics and operational efficiency.

### Current J0, J1, J2, J3, J4, J5:

- Refers to the electrical current drawn by each joint motor.
- Provides information on the effort each motor is exerting.
- Useful for detecting anomalies in power consumption or potential mechanical failures.

In the UR3 cobot, typically:
- J0: Rotating base
- J1: "Shoulder" - first main joint
- J2: "Elbow" - second main joint
- J3: First wrist joint
- J4: Second wrist joint – usually allows rotation
- J5: Third wrist joint – typically enables final end effector rotation

---

### Clustering Graphs (K-Means, Hierarchical, and DBSCAN):
- **X-axis**: PCA Component 1
- **Y-axis**: PCA Component 2

**Interpretation**: These values don't represent "better" or "worse" scenarios. They are simply coordinates in a 2D space representing the two main features extracted from the original data. Points closer together in this space are more similar in the original dataset.

---

### Elbow Method Plot:
- **X-axis**: Number of clusters
- **Y-axis**: SSE (Sum of Squared Errors)

**Interpretation**: Lower SSE values are generally better, as they indicate less variation within clusters. However, the goal is to find a "knee" in the curve, where increasing the number of clusters doesn't significantly reduce SSE. This inflection point suggests the optimal number of clusters.

---

### DBSCAN K-Distance Graph:
- **X-axis**: Data points sorted by distance
- **Y-axis**: Epsilon (distance)

**Interpretation**: There is no "best" value here. The goal is to identify a "knee" in the curve, similar to the elbow method. This knee suggests a good epsilon value for DBSCAN, which is the maximum distance between two samples for one to be considered in the neighborhood of the other.

---

### Time Series Plot:
- **X-axis**: Index (time)
- **Y-axis**: Variable value (temperature or current)

**Interpretation**:
- **For Temperatures**: Generally, lower values are better, as higher temperatures can indicate overheating or inefficiency.
- **For Currents**: Interpretation depends on the robot’s specific context. Consistent values within the expected range are generally good. Spikes or very high values may indicate a problem.

---

## Author

This repository was developed as part of the **Análisis de datos** course by **Julio Lara**, from the **Licenciatura en Ingeniería de Sistemas y Computación** career at the **Universidad Tecnológica de Panamá (UTP)**.

---

> **Note**: This project was originally developed in Spanish. The README is written in English for documentation and sharing purposes.


