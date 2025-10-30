# Load necessary libraries
rm(list=ls())
suppressMessages(library(optparse))
suppressMessages(library(tidyverse))
suppressMessages(library(data.table))
suppressMessages(library(caret))
suppressMessages(library(Boruta))
suppressMessages(library(ImageGP))
suppressMessages(library(ggplot2))
suppressMessages(library(ggpubr))
suppressMessages(library(doParallel))
suppressMessages(library(pROC))

option_list <- list(
  make_option("--filein",  action = "store"),
  make_option("--sampleinfo",  action = "store"),
  make_option("--fileouttrain",  action = "store"),
  make_option("--fileouttest",  action = "store")
)
opt = parse_args(OptionParser(option_list = option_list))

# step1: Data preprocessing
long.df <- fread(opt$filein, header = TRUE) %>% 
  as.data.frame()
# filein format
# id    seqnames    short    middle    long    nfrags
# sample1    chr1    13312    66908    8074    88294

# Function to process features
process_feature <- function(df, feature_type) {
  df %>%
    ungroup() %>%
    dplyr::select(!!sym(feature_type), id, bin) %>%
    spread(id, !!sym(feature_type)) %>%
    dplyr::select(-bin) %>%
    # na.omit() %>%
    apply(2, function(x) (x / sum(x)) * 2608) %>%
    t() %>%
    as.data.frame() %>%
    rownames_to_column("id") %>%
    setNames(c("id", paste0(feature_type, "_num", 1:2608)))
}

# Process features for frag, short, middle, and long
feature_types <- c("nfrags", "short", "middle", "long")
fragment.df.list <- map(feature_types, ~process_feature(long.df, .))

# Combine data frames by 'id'
df.frags <- Reduce(function(x, y) left_join(x, y, by = "id"), fragment.df.list)

# Read sample information
sample_466 <- read.table(opt$sampleinfo, header=T, sep="\t") %>%
  dplyr::rename(id=SampleID, sample_type=Type, cohort=Cohort) %>%
  dplyr::select(c(id, sample_type, cohort))

# Merge sample information with feature data
data <- merge(sample_466, df.frags, by="id")

data$sample_type <- ifelse(data$sample_type=="Healthy", "Healthy", "Tumor")


# Function to filter and select columns
process_data <- function(df, data_set, feature.type) {
  Set <- df %>% 
    dplyr::filter(cohort == data_set) %>% 
    dplyr::select(sample_type, matches(feature.type)) %>%
    mutate(sample_type = as.factor(sample_type))
  return(Set)
}

# Apply the function to create train and test sets
train <- process_data(data, "TrainSet", "num")
test <- process_data(data, "TestSet", "num")


train.pca <- train.set %>% dplyr::select(-c(sample_name))
test.pca <- test.set %>% dplyr::select(-c(sample_name))  

rownames(train.pca) <- train.set$sample_name
rownames(test.pca) <- test.set$sample_name

pca.model <- prcomp(train.pca[, -1], center=T, scale=T)
va.explained <- pca.model$sdev^2 / sum(pca.model$sdev^2)
cum.var.explained <- cumsum(va.explained)
num.components <- which(cum.var.explained >= 0.98)[1]
train.pc <- pca.model$x %>% as.data.frame() %>% rownames_to_column(var="sample_name")
test.pc <- predict(pca.model, newdata = test.pca[, -1]) %>% as.data.frame() %>% 
  rownames_to_column(var = "sample_name")

train.pcSet <- merge(train.set[, c("sample_name", "sample_type")], train.pc, by="sample_name")
test.pcSet <- merge(test.set[, c("sample_name", "sample_type")], test.pc, by="sample_name")
pc.var <- paste0("PC", seq(1, 200, 1))
train <- train.pcSet %>% dplyr::select(c(sample_name, sample_type, pc.var[1:num.components]))
test <- test.pcSet %>% dplyr::select(c(sample_name, sample_type, pc.var[1:num.components]))

# step3: Train the model
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                        allowParallel = T, summaryFunction = twoClassSummary)

# Grid of hyperparameter values
hyper_grid <- expand.grid(
  alpha = seq(0.1, 1, by = 0.1),        # Example: Sequence of values for alpha
  lambda = seq(0.1, 1, by = 0.1)  # Example: Sequence of values for lambda
)

# Grid search using the train function
set.seed(123)
cl <- makePSOCKcluster(15)
registerDoParallel(cl)

m_glmnet <- train(
  sample_type ~ ., data = train[, -c(1)], method = "glmnet", metric = "Kappa",
  tuneGrid = hyper_grid, trControl = control)

stopCluster(cl)

best.lambda <- m_glmnet$bestTune$lambda

raw.pred.train <- predict(m_glmnet, train, type = "raw")
mat.train <- confusionMatrix(raw.pred.train, train$sample_type, positive='Tumor')
print(mat.train)

raw.pred.test <- predict(m_glmnet, test, type = "raw")
mat.test <- confusionMatrix(raw.pred.test, test$sample_type, positive='Tumor')
print(mat.test)