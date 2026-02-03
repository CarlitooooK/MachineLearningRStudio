# 1. Carga de librerías y datos
if(!require(C50)) install.packages("C50")
if(!require(caret)) install.packages("caret")
if(!require(ggplot2)) install.packages("ggplot2")
library(C50)
library(caret)
library(ggplot2)

heart_data <- read.csv("heart-2.csv")
heart_data$target <- as.factor(heart_data$target)


ggplot(heart_data, aes(x = age, fill = target)) +
  geom_histogram(binwidth = 5, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("steelblue", "orange"), labels = c("Sano", "Enfermo")) +
  labs(title = "Distribución de Edad por Target", x = "Edad", y = "Frecuencia") +
  theme_minimal()

set.seed(123)
train_idx <- createDataPartition(heart_data$target, p = 0.8, list = FALSE)
train_set <- heart_data[train_idx, ]
test_set <- heart_data[-train_idx, ]

# Aplicación de poda
control_poda <- C5.0Control(CF = 0.1, minCases = 10)
modelo <- C5.0(target ~ ., data = train_set, control = control_poda)


predicciones <- predict(modelo, test_set)
mc <- confusionMatrix(predicciones, test_set$target)

#Falsos Positivos y Negativos
mc_df <- as.data.frame(mc$table)

ggplot(mc_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
  geom_text(aes(label = Freq), size = 8, color = "black") +
  labs(title = "Visualización de Errores de Clasificación",
       subtitle = paste("Precisión:", round(mc$overall['Accuracy'], 3)),
       x = "Referencia (Real)", y = "Predicción (Modelo)") +
  annotate("text", x = 1, y = 2, label = "Falsos Negativos", color = "red", fontface = "bold") +
  annotate("text", x = 2, y = 1, label = "Falsos Positivos", color = "red", fontface = "bold") +
  theme_light()

plot(modelo, main = "Estructura del Árbol de Decisión Podado")

