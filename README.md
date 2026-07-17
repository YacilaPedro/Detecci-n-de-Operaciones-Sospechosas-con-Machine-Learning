# Deteccion-de-Operaciones-Sospechosas 
Proyecto tipo datathon desarrollado en el curso de Machine Learning para clasificar operaciones como sospechosas o no sospechosas. Se aplicó preprocesamiento, balanceo, selección de variables y un ensamble heterogéneo evaluado con F1-Score.

## Contexto del proyecto

Este proyecto fue desarrollado como parte de una evaluación tipo datathon del curso de **Machine Learning**, dictado por el profesor **Salinas**. El objetivo principal fue construir un modelo predictivo para identificar posibles casos de lavado, utilizando como variable objetivo `LAVADO`, con las clases `No_sospechoso` y `Sospechoso`.

La evaluación se centró en desarrollar un flujo completo de modelamiento supervisado, respetando estrictamente las reglas indicadas por el docente. La métrica principal de evaluación fue el **F1-Score**, debido al fuerte desbalance de clases presente en la base de datos, donde la clase de interés (`Sospechoso`) representaba una proporción muy pequeña del total de observaciones.

## Reglas de la evaluación

Para esta evaluación, el docente estableció una serie de restricciones metodológicas y técnicas que debían cumplirse obligatoriamente:

- El trabajo debía realizarse de forma individual.
- El código debía poder ejecutarse directamente en R/RStudio.
- No se permitía el uso de internet durante la evaluación.
- Solo se podían utilizar paquetes, funciones y algoritmos desarrollados en clase.
- No estaba permitido usar estructuras como `for`, `while`, funciones creadas por el usuario, `View()`, `install.packages()` ni comandos no solicitados.
- La partición de los datos debía realizarse en 70% para entrenamiento y 30% para prueba.
- En caso de aplicar balanceo de clases, solo se podía utilizar un único algoritmo de balanceo.
- Para selección de variables, se permitía el uso de métodos revisados en clase, como Boruta o Random Forest.
- El modelo final debía ser un ensamble heterogéneo conformado exactamente por cinco algoritmos de clasificación.
- No estaba permitido usar stacking, voto mayoritario ni otros métodos de combinación distintos al promedio simple de probabilidades.
- El umbral de clasificación debía ser 0.5.
- La última línea del script debía devolver el indicador solicitado:

```r
modelo$byClass["F1"]
```

## Metodología aplicada

El flujo de trabajo seguido fue el siguiente:

1. Carga de la base de datos.
2. Definición de la variable objetivo `LAVADO`.
3. Tratamiento de variables categóricas y numéricas.
4. Imputación de valores faltantes.
5. Partición estratificada de los datos en entrenamiento y prueba.
6. Selección de variables mediante **Boruta**.
7. Balanceo de la clase minoritaria mediante **SMOTE**.
8. Entrenamiento de cinco algoritmos de clasificación.
9. Promedio simple de probabilidades.
10. Clasificación final usando un umbral de `0.5`.
11. Evaluación mediante **F1-Score**.

## Ensamble heterogéneo

El modelo final se construyó como un **ensamble heterogéneo**, combinando algoritmos de distintas familias de aprendizaje supervisado. La combinación final se realizó mediante promedio aritmético simple de las probabilidades estimadas por cada modelo.

Los cinco modelos utilizados fueron:

- Regresión logística
- Gradient Boosting Machine
- Support Vector Machine con kernel radial
- k vecinos más cercanos
- Red neuronal

## Métrica de evaluación

La métrica principal fue el **F1-Score**, debido a que la variable objetivo presentaba un fuerte desbalance entre las clases `No_sospechoso` y `Sospechoso`.

El **F1-Score** permite equilibrar la precisión y la sensibilidad, siendo especialmente útil cuando la clase positiva es minoritaria.

## Herramientas utilizadas

- R
- RStudio
- caret
- Boruta
- performanceEstimation
- VIM
- gbm
- nnet

## Conclusión

Este proyecto permitió aplicar un flujo completo de **Machine Learning** bajo restricciones reales de evaluación, priorizando la correcta ejecución del código, el cumplimiento metodológico y el desempeño predictivo sobre una clase minoritaria.


## Resultado 

En la versión mejorada del modelo, se obtuvo el siguiente desempeño en el conjunto de prueba:

- Sensitivity: 0.75
- Precision: 0.8571
- Specificity: 0.9989
- Balanced Accuracy: 0.8744
- F1-Score: 0.80

El modelo logró detectar el 75% de los casos sospechosos, manteniendo una alta precisión y una baja tasa de falsos positivos. Debido al fuerte desbalance de clases, los resultados deben interpretarse con cautela, ya que pequeñas variaciones en la partición pueden afectar la métrica final.



























