concrete <- read.csv("concrete.csv")

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))

concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

library(neuralnet)

set.seed(12345)
concrete_model <- neuralnet(
  strength ~ cement + slag + ash + water + superplastic +
    coarseagg + fineagg + age,
  data = concrete_train
)

plot(concrete_model)

model_results <- compute(concrete_model, concrete_test[1:8])
predicted_strength <- model_results$net.result
cor(predicted_strength, concrete_test$strength)

set.seed(12345)
concrete_model2 <- neuralnet(
  strength ~ cement + slag + ash + water + superplastic +
    coarseagg + fineagg + age,
  data = concrete_train,
  hidden = 5
)

plot(concrete_model2)

model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

softplus <- function(x) {
  log(1 + exp(x))
}

set.seed(12345)
concrete_model3 <- neuralnet(
  strength ~ cement + slag + ash + water + superplastic +
    coarseagg + fineagg + age,
  data = concrete_train,
  hidden = c(5, 5),
  act.fct = softplus
)

plot(concrete_model3)

model_results3 <- compute(concrete_model3, concrete_test[1:8])
predicted_strength3 <- model_results3$net.result
cor(predicted_strength3, concrete_test$strength)

strengths <- data.frame(
  actual = concrete$strength[774:1030],
  pred = predicted_strength3
)

unnormalize <- function(x) {
  x * (max(concrete$strength) - min(concrete$strength)) +
    min(concrete$strength)
}

strengths$pred_new <- unnormalize(strengths$pred)
strengths$error_pct <- (strengths$pred_new - strengths$actual) / strengths$actual

cor(strengths$pred_new, strengths$actual)
cor(strengths$pred, concrete_test$strength)

