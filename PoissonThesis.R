Elopoisson <- read_csv("Desktop/ThesisGit/Elopoisson.csv")
Elopoisson<-Elopoisson[ which(Elopoisson$Year!=2016 ), ]

Elopoisson$EffHf<-(Elopoisson$HF*Elopoisson$HomeAway)/400
Elopoisson$RatingDif<-(Elopoisson$Rating-Elopoisson$OppRating)/400

Eloreg<-glm(Elopoisson$Goals~Elopoisson$RatingDif+Elopoisson$EffHf, family="poisson")
summary(Eloreg)

ManagerPoisson <- read_csv("Desktop/ThesisGit/ManagerPoisson.csv")
ManagerPoisson<-ManagerPoisson[ which(ManagerPoisson$Year!=2016 ), ]

ManagerPoisson$RatingDif<-ManagerPoisson$RatingDif/400
ManagerPoisson$EffHf<-ManagerPoisson$EffHf/400

Managerreg<-glm(ManagerPoisson$Goals~ManagerPoisson$EffHf+ManagerPoisson$RatingDif, family="poisson")
summary(Managerreg)


BoostPoisson <- read_csv("Desktop/ThesisGit/BoostPoisson.csv")
BoostPoisson<-BoostPoisson[ which(BoostPoisson$Year!=2016 ), ]

BoostPoisson$RatingDif<-BoostPoisson$RatingDif/400
BoostPoisson$EffHf<-BoostPoisson$EffHf/400
Boostreg<-glm(BoostPoisson$Goals~BoostPoisson$EffHf+BoostPoisson$RatingDif, family="poisson")
summary(Boostreg)
