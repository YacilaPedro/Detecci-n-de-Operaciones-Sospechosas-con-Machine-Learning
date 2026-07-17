options(digits = 10)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

semilla <- 1969
set.seed(semilla)

# 0. PAQUETES
library(pacman)
p_load(caret, VIM, dplyr, Boruta, performanceEstimation,
       gbm, nnet)

# 1. PREPARACION Y PREPROCESAMIENTO DE DATOS

datos <- read.csv("dataset_lavado.csv", stringsAsFactors = FALSE)

# Correccion de inconsistencias en la variable objetivo
datos$LAVADO[datos$LAVADO == "No sospechoso"] <- "No_sospechoso"

# Correccion de inconsistencias en variables categoricas
datos$jurisdiccion_riesgo[datos$jurisdiccion_riesgo == "Bja"] <- "Baja"

# Conversion de variables categoricas a factor
datos$canal_transaccion   <- as.factor(datos$canal_transaccion)
datos$tipo_cliente        <- as.factor(datos$tipo_cliente)
datos$jurisdiccion_riesgo <- as.factor(datos$jurisdiccion_riesgo)
datos$pep                 <- as.factor(datos$pep)
datos$alerta_manual       <- as.factor(datos$alerta_manual)
datos$perfil_operacional  <- as.factor(datos$perfil_operacional)

# Configuracion de la variable objetivo
datos$LAVADO <- factor(datos$LAVADO,
                       levels = c("No_sospechoso", "Sospechoso"))

target <- "LAVADO"
neg <- "No_sospechoso"
pos <- "Sospechoso"

# Verificacion inicial de clases
table(datos[[target]])
colSums(is.na(datos))

# Imputacion kNN solo sobre predictores con valores faltantes
# No se imputa la variable objetivo LAVADO
predictores <- datos[, setdiff(names(datos), target)]

set.seed(semilla)
predictores_imp <- VIM::kNN(predictores,
                            variable = c("canal_transaccion",
                                         "indice_riesgo_cliente"),
                            imp_var = FALSE)

datos_imputados <- cbind(predictores_imp,
                         LAVADO = datos[[target]])

datos_imputados$LAVADO <- factor(datos_imputados$LAVADO,
                                 levels = c(neg, pos))

# Verificacion posterior a imputacion
colSums(is.na(datos_imputados))
table(datos_imputados[[target]])

# 2. PARTICION ESTRATIFICADA 70/30

set.seed(semilla)
train_index <- createDataPartition(datos_imputados[[target]],
                                   p = 0.70,
                                   list = FALSE)

train <- datos_imputados[train_index, ]
test  <- datos_imputados[-train_index, ]

# Verificar proporcion de clases
table(train[[target]])
table(test[[target]])
prop.table(table(train[[target]]))
prop.table(table(test[[target]]))

# 3. BALANCEO DE DATOS EN TRAINING
# Unico algoritmo de balanceo utilizado: SMOTE

formula_base <- as.formula(paste(target, "~ ."))

set.seed(semilla)
train_smote <- performanceEstimation::smote(formula_base,
                                            data = train,
                                            perc.over = 13,
                                            perc.under = 10.25)

train_smote[[target]] <- factor(train_smote[[target]],
                                levels = c(neg, pos))

table(train_smote[[target]])

# 4. SELECCION DE VARIABLES CON BORUTA
# Se aplica sobre el conjunto de entrenamiento balanceado

set.seed(semilla)
boruta_obj <- Boruta(formula_base,
                     data = train_smote,
                     doTrace = 0)

boruta_obj <- TentativeRoughFix(boruta_obj)

vars_confirmadas <- getSelectedAttributes(boruta_obj,
                                          withTentative = FALSE)

vars_confirmadas

formula_modelo <- as.formula(paste(target, "~",
                                   paste(vars_confirmadas, collapse = " + ")))

train_final <- train_smote[, c(vars_confirmadas, target)]
test_final  <- test[, c(vars_confirmadas, target)]

train_final[[target]] <- factor(train_final[[target]], levels = c(neg, pos))
test_final[[target]]  <- factor(test_final[[target]], levels = c(neg, pos))

# 5. MODELADO: 5 ALGORITMOS DE CLASIFICACION

ctrl <- trainControl(method = "cv",
                     number = 3,
                     classProbs = TRUE)

# a. Regresion Logistica
set.seed(semilla)
mod_glm <- train(formula_modelo,
                 data = train_final,
                 method = "glm",
                 trControl = ctrl)

# b. Gradient Boosting Machine
set.seed(semilla)
mod_gbm <- train(formula_modelo,
                 data = train_final,
                 method = "gbm",
                 trControl = ctrl,
                 verbose = FALSE)

# c. Maquina de Soporte Vectorial con Kernel Radial
set.seed(semilla)
mod_svm <- train(formula_modelo,
                 data = train_final,
                 method = "svmRadial",
                 trControl = ctrl)

# d. K vecinos mas cercanos
set.seed(semilla)
mod_knn <- train(formula_modelo,
                 data = train_final,
                 method = "knn",
                 trControl = ctrl)

# e. Red neuronal
set.seed(semilla)
mod_nnet <- train(formula_modelo,
                  data = train_final,
                  method = "nnet",
                  trControl = ctrl,
                  trace = FALSE)

# 6. ENSAMBLE HETEROGENEO POR PROMEDIO SIMPLE DE PROBABILIDADES

prob_glm  <- predict(mod_glm,  newdata = test_final, type = "prob")[, pos]
prob_gbm  <- predict(mod_gbm,  newdata = test_final, type = "prob")[, pos]
prob_svm  <- predict(mod_svm,  newdata = test_final, type = "prob")[, pos]
prob_knn  <- predict(mod_knn,  newdata = test_final, type = "prob")[, pos]
prob_nnet <- predict(mod_nnet, newdata = test_final, type = "prob")[, pos]

# Promedio aritmetico simple
prob_promedio <- (prob_glm + prob_gbm + prob_svm + prob_knn + prob_nnet) / 5

# Clasificacion final con umbral 0.5
prediccion_final <- factor(ifelse(prob_promedio > 0.5, pos, neg),
                           levels = c(neg, pos))

# 7. EVALUACION DEL MODELO

modelo <- caret::confusionMatrix(prediccion_final,
                                 test_final[[target]],
                                 positive = pos)

modelo
modelo$byClass["Sensitivity"]
modelo$byClass["Precision"]
modelo$byClass["Specificity"]
modelo$byClass["Balanced Accuracy"]
modelo$byClass["F1"]

# COMENTARIO
# El ensamble heterogeneo esta conformado por cinco algoritmos:
# 1. glm       : Regresion Logistica
# 2. gbm       : Gradient Boosting Machine
# 3. svmRadial : Maquina de Soporte Vectorial con Kernel Radial
# 4. knn       : K vecinos mas cercanos
# 5. nnet      : Red neuronal artificial
#
# La combinacion final se realiza mediante promedio simple de probabilidades.
# No se utiliza stacking, voto mayoritario ni umbral optimizado.
# El umbral utilizado es 0.5.
