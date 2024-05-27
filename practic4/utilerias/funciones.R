# Funciones para ayuda de la transicion
multi.hist <- function(df) {nvar <- dim(df)[2]
for (i in 1:nvar) {
  name=names(df)[i]
  x <- df[,i] 
  hist(x, freq = FALSE,
       main="Histograma vs Kernel", ylab="Frecuencia Relativa", xlab=name, 
       border="white", cex=0.4, cex.main=1, col="azure3",
       bty = "n");grid()
  
  mu <- mean(x); s <- sd(x)
  lines( density(x, na.rm = T ) )
  curve( dnorm(x, mu, s), add=TRUE, col="darkgray" )
  
  legend("topright", 
         legend = c("Densidad Normal","Densidad Kernel"), 
         col = c("darkgray","black"),
         pch = c(1,12), bty = "n", pt.cex = 1, cex = 1, text.col = "black", horiz = F, 
         inset = c(0.1, 0.1))
  
  # Boxplot
  par(new = TRUE)
  boxplot(x, horizontal = TRUE, axes = F, border = "black",
          lwd = 2, col = rgb(0, 1, 1, alpha = 0.15), cex=0.4)
}}

# Normalizacion Min-Max
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# identificar los tipos de datos
tipos <- function(df){
  tipoDatos <- sapply(df, class) # Saber los tipos de datos
  continuas <-  which(tipoDatos == "numeric") # continuas
  enteras <- which(tipoDatos == "integer") # enteras
  numericas <- c(continuas,enteras)
  
  nominales <- which( tipoDatos == "factor") # categoricas
  ordinales <- which( sapply(df, is.ordered) )  # ordinales
  
  fecha <- which(tipoDatos == "Date")
  categoricas <- c(nominales, ordinales, fecha)
  tipos <- list(numericas=numericas, categoricas= categoricas)
  return(tipos)
}

# Funcion para identificar variables altamente correlacionadas
altCorr <- function(df, nivel){
  correlaciones <- cor(df) # Matriz de correlaciones
  corAltas <- c()
  correlacionados <- data.frame(correlacion=numeric(), 
                                renglon= character(), 
                                columna=character())
  
  # Valores que tienen correlacion mayor a 0.9
  for( valor in unique(sort(correlaciones,decreasing = TRUE)) ){
    if(valor < nivel) break
    corAltas <- c(corAltas, valor)
  }
  
  # Referencias de variables con correlacion mayor a 0.9
  for( variable in colnames(correlaciones) ){
    for( valor in corAltas ){
      if(valor %in% correlaciones[, variable]){
        renglones <- which( correlaciones[, variable] == valor)
        renglones <- rownames(correlaciones)[renglones]
        m <- length(renglones)
        correlacionados <- rbind(correlacionados,
                                 data.frame(correlacion = rep(valor,m), 
                                            renglon = renglones, 
                                            columna = rep(variable,m) ))
      }
    }
  }
  
  # Eliminando redundancia sistematica
  correlacionados <- correlacionados[-which(correlacionados$renglon == correlacionados$columna),]
  
  for( n in 1:nrow(correlacionados) ){
    renglon <- as.character(correlacionados[n, c("renglon","columna")])
    renglon <- sort(renglon)
    correlacionados[n, c("renglon","columna")] <- list(renglon[1],renglon[2]) 
  }
  correlacionados <- correlacionados[!duplicated(correlacionados),]
  return(correlacionados)
}

# Funcion addstock geometrica
geometric <- function(xt, x_t, theta){
  return( xt + (theta * x_t) )
}

# Funcion Hill
hill <- function(xt, kappa, slope){
  return ( 1 / (1 + (xt / kappa)^(-slope) ) )
}