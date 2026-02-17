# --- Máquinas de Soporte Vectorial en R ---
# Autor: Abraham Sánchez López, 2026
# Grupo MOVIS, FCC-BUAP

# 1. Carga de librerías
library(e1071)
library(ROCR)
library(ISLR2)
library(plotly)
library(tibble)
library(caret)
library(ggplot2)
library(dplyr)

# 2. Ejemplo didáctico de expand.grid()
x <- 1:3
y_simple <- 4:6
expand.grid(x, y_simple)

# 3. Creación de datos sintéticos
set.seed(1)
x1 <- runif(30)
x2 <- runif(30)
y <- factor(sample(x = c(0, 1), size = 30, replace = TRUE, prob = c(0.6, 0.4)))
df1 <- data.frame(y, x1, x2)

# 4. Estimación del modelo SVM lineal
m1 <- svm(formula = y ~ x1 + x2,
          data = df1,
          kernel = "linear",
          cost = 10,
          scale = FALSE)

# 5. Visualización básica y preparación de predicciones
plot(m1, df1)
df1["fitted"] <- predict(object = m1, newdata = df1)

# Intento de gráfico con geom_tile (Figura 3)
df1 %>%
  ggplot() +
  geom_tile(aes(x = x2, y = x1, fill = fitted))

# Gráfico con geom_point (Figura 4)
df1 %>%
  ggplot() +
  geom_point(aes(x = x2, y = x1, color = fitted))

# Gráfico con hiperplano manual (Figura 5)
df1 %>%
  ggplot() +
  geom_point(aes(x = x2, y = x1, color = fitted)) +
  geom_segment(aes(x = 0, y = 0.32, xend = 1, yend = 0.425)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, NA)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, NA)) +
  theme_bw()

# 6. Creación de fondo coloreado con expand.grid()
set.seed(1)
y_grid <- factor(sample(x = c(0, 1), size = 2500, replace = TRUE))
df2 <- data.frame(expand.grid(
  x1 = seq(from = min(df1$x1), to = max(df1$x1), length.out = 50),
  x2 = seq(from = min(df1$x2), to = max(df1$x2), length.out = 50)),
  y = y_grid)

df2["fitted_color"] <- predict(object = m1, newdata = df2)

# 7. Identificación de Vectores de Soporte
m1$index
df1 <- df1 %>%
  rownames_to_column("row_idx") %>%
  mutate(index = ifelse(row_idx %in% m1$index, "support", "not_support"))

# Gráfico de clasificación SVM final (Figura 9)
df2 %>%
  ggplot(aes(x = x2, y = x1)) +
  geom_tile(aes(fill = fitted_color)) +
  geom_point(data = df1, aes(x = x2, y = x1, color = y, shape = index), size = 3) +
  scale_color_manual(values = c("black", "red")) +
  scale_shape_manual(values = c(1, 4)) +
  theme_void()

# 8. Aplicación con datos reales: barbecue.RData
load("barbecue.RData")

# Gráfico interactivo inicial
ggplotly(
  barbecue %>%
    ggplot(aes(x = chicken, y = beef, color = income, shape = income)) +
    geom_point() +
    labs(x = "chicken", y = "beef", shape = NULL) +
    scale_color_manual("Labels:", values = c("darkorchid", "orange")) +
    theme_bw()
)

# Modelo SVM para barbecue
svmfit_1 <- svm(formula = income ~ beef + chicken,
                data = barbecue,
                kernel = "linear",
                cost = 10,
                scale = FALSE)

plot(x = svmfit_1, data = barbecue) #

# Fondo para barbecue
background <- expand.grid(
  beef = seq(from = min(barbecue$beef), to = max(barbecue$beef), length.out = 150),
  chicken = seq(from = min(barbecue$chicken), to = max(barbecue$chicken), length.out = 150)
)
background["fitted_color"] <- predict(object = svmfit_1, newdata = background)

# Preparación de datos y vectores de soporte para barbecue
barbecue <- barbecue %>%
  rownames_to_column("row_idx") %>%
  mutate(index = ifelse(row_idx %in% svmfit_1$index, "support", "not_support"),
         fitted = predict(object = svmfit_1, data = .))

# Gráfico elegante (Figura 14)
background %>%
  ggplot(aes(x = chicken, y = beef)) +
  geom_tile(aes(fill = fitted_color)) +
  geom_point(data = barbecue, aes(x = chicken, y = beef, color = income, shape = index), size = 3) +
  scale_fill_manual(values = c("cornsilk2", "aliceblue")) +
  scale_color_manual(values = c("orange", "darkorchid")) +
  scale_shape_manual(values = c(1, 4)) +
  labs(x = "chicken", y = "beef", fill = "Predicted Category", 
       shape = "SV", color = "Observed Category") +
  theme(panel.background = element_rect(fill = NA),
        panel.border = element_rect(color = "black", fill = NA),
        legend.position = "bottom",
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8))

# 9. Límite no lineal y Kernels
set.seed(1)
x_nl <- matrix(rnorm(200 * 2), ncol = 2)
x_nl[1:100, ] <- x_nl[1:100, ] + 2
x_nl[101:150, ] <- x_nl[101:150, ] - 2
y_nl <- c(rep(1, 150), rep(2, 50))
dt <- data.frame(x = x_nl, y = as.factor(y_nl))
plot(x_nl[, 1], x_nl[, 2], pch = 16, col = y_nl * 2)

# SVM Polinomial
svmfit_poly <- svm(y ~ ., data = dt, kernel = "polynomial", cost = 1, degree = 2)
plot(svmfit_poly, dt, grid = 200, col = c("pink", "lightblue"))

# Tuning con Kernel Radial
tune.out <- tune(svm, y ~ ., data = dt, kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100),
                               gamma = c(0.5, 1, 2, 3, 4)))
plot(tune.out$best.model, dt, grid = 200, col = c("pink", "lightblue"))

# 10. Curva ROC
set.seed(1)
train <- sort(sample(200, 100), decreasing = TRUE)
model_radial <- svm(y ~ ., data = dt[train, ], kernel = "radial", cost = 1, gamma = 0.5)
fit_decision <- attributes(predict(model_radial, dt[-train, ], decision.values = TRUE))$decision.values

pred_rocr <- prediction(fit_decision, dt[-train, "y"])
auc_ROCR <- performance(pred_rocr, measure = "auc")
auc_ROCR@y.values[[1]]