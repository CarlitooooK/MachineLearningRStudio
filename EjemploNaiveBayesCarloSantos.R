library(e1071)
library(caret)

misDatos <- read.table("nazi.txt", header = TRUE)

datos <- misDatos[rep(row.names(misDatos), misDatos$Co),
                  c("R", "C", "Re", "G", "M")]
rownames(datos) <- NULL

names(datos) <- c("Religion", "Cohorte", "Residencia", "Genero", "Afiliacion")

datos$Religion <- factor(datos$Religion, levels=c(1,2,3),
                         labels=c("Protestante","Catolica","Ninguna"))

datos$Cohorte <- factor(datos$Cohorte, levels=c(1,2,3,4,5),
                        labels=c("Imperio","Imperio_Tardio",
                                 "Weimar_Temprano",
                                 "Weimar_Tardio",
                                 "Tercer_Reich"))

datos$Residencia <- factor(datos$Residencia, levels=c(1,2),
                           labels=c("Rural","Urbana"))

datos$Genero <- factor(datos$Genero, levels=c(1,2),
                       labels=c("Masculino","Femenino"))

datos$Afiliacion <- factor(datos$Afiliacion, levels=c(0,1),
                           labels=c("No","Si"))

set.seed(123)
idx <- createDataPartition(datos$Afiliacion, p=0.7, list=FALSE)
train <- datos[idx, ]
test  <- datos[-idx, ]

train_bal <- upSample(
  x = train[, c("Religion","Cohorte","Residencia","Genero")],
  y = train$Afiliacion
)
names(train_bal)[5] <- "Afiliacion"

modelo_nb <- naiveBayes(
  Afiliacion ~ Religion + Cohorte + Residencia + Genero,
  data = train_bal,
  laplace = 1
)

prob_test <- predict(modelo_nb, test, type="raw")[,"Si"]

umbral <- 0.35

pred_final <- factor(
  ifelse(prob_test >= umbral, "Si", "No"),
  levels = c("No","Si")
)

cm <- confusionMatrix(
  pred_final,
  test$Afiliacion,
  positive = "Si"
)

print(cm)

par(mfrow=c(1,2))

hist(prob_test[test$Afiliacion=="No"],
     main="Probabilidades (No afiliados)",
     xlab="P(Afiliación = Sí)",
     col="lightgreen", breaks=20, xlim=c(0,1))
abline(v=umbral, col="red", lwd=2, lty=2)

hist(prob_test[test$Afiliacion=="Si"],
     main="Probabilidades (Afiliados)",
     xlab="P(Afiliación = Sí)",
     col="salmon", breaks=20, xlim=c(0,1))
abline(v=umbral, col="red", lwd=2, lty=2)

par(mfrow=c(1,1))

