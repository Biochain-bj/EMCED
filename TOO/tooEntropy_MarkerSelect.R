suppressMessages(library(optparse))
suppressMessages(library(Boruta))
suppressMessages(library(tidyverse))
suppressMessages(library(dplyr))

option_list <- list(
  make_option("--filein",  action = "store"),
  make_option("--fileout",  action = "store")
)
opt = parse_args(OptionParser(option_list = option_list))

set.seed(1234)

data <- read.table(opt$filein,header = T,sep='\t',check.names = 0)
# filein format:
# SampleID    Type    Cohort    feature1    feature2    ...    featuren    Sex
# Sample1    COREAD    TrainSet    0.9281437125748504    0.9233716475095786    ...    0.9106840022611644    Female
data$Type <- as.factor(data$Type)
boruta_stat <- Boruta(Type ~ ., data = data %>% filter(Cohort=="TrainSet",Type!="Healthy") %>% select(-c(SampleID,Cohort)), doTrace = 2, maxRuns = 500) %>% attStats() %>% arrange(desc(meanImp))
boruta_stat <- tibble::rownames_to_column(boruta_stat, "marker_index")
write.table(boruta_stat,opt$fileout,row.names = F,sep='\t',quote=FALSE)
