########################### Plots #################################
library(tidyverse)

##Bivariate plots
train_dummy %>%
  group_by(target1) %>%
  ggplot() +
  geom_jitter(data = train_dummy %>% 
                group_by(target1),
              aes(x = target1, y = YearsInCurrentRole, color = factor(target1)), 
              position = position_jitter(w = 0.3, h = 0),
              alpha = 0.5) +
  geom_point(data = train_dummy %>% 
               group_by(target1),
             aes(x = target1, y = YearsInCurrentRole, color = factor(target1)),
             size = 5, alpha = 0.2) +
  labs(x = "Extra Hours(0/1)",
       y = "YearsInCurrentRole",
       title = "YearsInCurrentRole and Extra Hours")


### Univariate Plots
train_dummy %>%
  ggplot(aes(x = YearsInCurrentRole)) +
  geom_histogram()
  

##Q-Q plot
qqnorm(train_dummy$Attr27)
qqline(train_dummy$Attr27)  
  
