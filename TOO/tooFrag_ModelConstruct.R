# Load necessary libraries
rm(list=ls())
suppressMessages(library(optparse))
suppressMessages(library(tidyverse))
suppressMessages(library(data.table))
suppressMessages(library(caret))
suppressMessages(library(ImageGP))
suppressMessages(library(ggplot2))
suppressMessages(library(ggpubr))
suppressMessages(library(pheatmap))
suppressMessages(library(doParallel))

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
    rownames_to_column("sample_name") %>%
    setNames(c("sample_name", paste0(feature_type, "_num", 1:2608)))
}

# Process features for frag, short, middle, and long
feature_types <- c("nfrags", "short", "middle", "long")
fragment.df.list <- map(feature_types, ~process_feature(long.df, .))

# Combine data frames by 'id'
df.frags <- Reduce(function(x, y) left_join(x, y, by = "sample_name"), fragment.df.list)

# Read sample information
info <- read.table(opt$sampleinfo, header=T, sep="\t") %>%
  dplyr::select(c(sample_name, sample_type, cohort, gender)) %>% 
  dplyr::filter(sample_type!="Healthy")

# Merge sample information with feature data
data <- merge(info, df.frags, by="sample_name")

data$gender <- ifelse(data$gender=="Male", 1, 0) %>% as.factor()

# Function to filter and select columns
process_data <- function(df, data_set, feature.type) {
  Set <- df %>% 
    dplyr::filter(cohort == data_set) %>% 
    dplyr::select(sample_name, sample_type, gender, matches(feature.type)) %>%
    dplyr::mutate(sample_type = as.factor(sample_type))
  return(Set)
}

# Apply the function to create train and test sets
train.set <- process_data(data, "TrainSet", "nfrags")
test.set <- process_data(data, "TestSet", "nfrags")

## PCA
train.pca <- train.set %>% dplyr::select(-c(sample_name, gender))
test.pca <- test.set %>% dplyr::select(-c(sample_name, gender))

rownames(train.pca) <- train.set$sample_name
rownames(test.pca) <- test.set$sample_name

# pca模型
pca.model <- prcomp(train.pca[, -1], center=T, scale=T)

va.explained <- pca.model$sdev^2 / sum(pca.model$sdev^2)

# 计算累计方差比例
cum.var.explained <- cumsum(va.explained)

# 确定解释90%方差所需的主成分数量
num.components <- which(cum.var.explained >= 0.98)[1]

train.pc <- pca.model$x %>% as.data.frame() %>% rownames_to_column(var="sample_name")

# pca预测
test.pc <- predict(pca.model, newdata = test.pca[, -1]) %>% as.data.frame() %>% 
  rownames_to_column(var = "sample_name")


train.pcSet <- merge(train.set[, c("sample_name", "gender", "sample_type")], train.pc, by="sample_name")
test.pcSet <- merge(test.set[, c("sample_name", "gender", "sample_type")], test.pc, by="sample_name")
  
pc.var <- paste0("PC", seq(1, 200, 1))

train <- train.pcSet %>% dplyr::select(c(sample_name, sample_type, gender, pc.var[1:num.components]))
test <- test.pcSet %>% dplyr::select(c(sample_name, sample_type, gender, pc.var[1:num.components]))

# 模型训练

# Grid search using the train function
control <- trainControl(method="cv", number=5, classProbs= TRUE, 
                        summaryFunction = multiClassSummary)

# Grid of hyperparameter values
hyper_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),        # Example: Sequence of values for alpha
  lambda = seq(0, 1, by = 0.1)  # Example: Sequence of values for lambda
)

set.seed(123)
cl <- makePSOCKcluster(50)
registerDoParallel(cl)

set.seed(123)

m_glmnet <- train(
  sample_type ~ ., data = train[, -c(1)], method = "glmnet", metric = "Kappa",
  tuneGrid = hyper_grid, trControl = control,maxit = 50000) 

stopCluster(cl)
  
# Predictions
raw.pred.train <- predict(m_glmnet, train, type = "raw")

raw.pred.train <- factor(raw.pred.train, levels = c("Lung", "COREAD", "STAD",
                                                    "LIHC", "ESCA", "THCA",
                                                    "OV"))

actual.label <- factor(train$sample_type, levels = c("Lung", "COREAD", "STAD",
                                                    "LIHC", "ESCA", "THCA",
                                                    "OV"))

mat.train <- confusionMatrix(raw.pred.train, actual.label)

raw.pred.test <- predict(m_glmnet, test, type = "raw")
raw.pred.test <- factor(raw.pred.test, levels = c("Lung", "COREAD", "STAD",
                                                    "LIHC", "ESCA", "THCA",
                                                    "OV"))
actual.label <- factor(test$sample_type, levels = c("Lung", "COREAD", "STAD",
                                                    "LIHC", "ESCA", "THCA",
                                                    "OV"))

mat.test <- confusionMatrix(raw.pred.test, actual.label)

# Top-1 and Top-2 calculation
train.pred.prob <- predict(m_glmnet, train, type = "prob")

train.pred.res <- train.pred.prob %>%
  mutate(top1 = apply(train.pred.prob, 1, function(x) {
    sorted_indices <- order(x, decreasing = TRUE)
    names(x)[sorted_indices[1]]}),
    top2 = apply(train.pred.prob, 1, function(x) {
      sorted_indices <- order(x, decreasing = TRUE)
      names(x)[sorted_indices[2]]})) %>%
  mutate(gender = train$gender, true.label = train$sample_type) %>%
  mutate(true_equal_top1=(true.label==top1),
         true_equal_top2=(true.label==top2),
         top12=true_equal_top1 + true_equal_top2)

train.top1.acc <- sum(train.pred.res$true_equal_top1) / nrow(train)
train.top2.acc <- sum(train.pred.res$top12) / nrow(train)

test.pred.prob <- predict(m_glmnet, test, type = "prob")

test.pred.res <- test.pred.prob %>%
  mutate(
    top1 = apply(test.pred.prob, 1, function(x) {
      sorted_indices <- order(x, decreasing = TRUE)
      names(x)[sorted_indices[1]]}),
    top2 = apply(test.pred.prob, 1, function(x) {
      sorted_indices <- order(x, decreasing = TRUE)
      names(x)[sorted_indices[2]]})) %>%
  mutate(gender = test$gender, true.label = test$sample_type) %>%

  mutate(true_equal_top1=(true.label==top1),
         true_equal_top2=(true.label==top2),
         top12=true_equal_top1 + true_equal_top2)

test.top1.acc <- sum(test.pred.res$true_equal_top1) / nrow(test)
test.top2.acc <- sum(test.pred.res$top12) / nrow(test)

acc <- data.frame(train.acc = c(train.top1.acc, train.top2.acc), 
                  test.acc = c(test.top1.acc, test.top2.acc))

train.pred.res <- cbind(sample_name=train$sample_name, train.pred.res)
test.pred.res <- cbind(sample_name=test$sample_name, test.pred.res)

print(acc)
print(mat.test)

confusion.mat <- list(train.mat = mat.train$table, test.mat = mat.test$table)

frag.res <- list(model=m_glmnet, confirmed.var=var, acc=acc, confusion.mat=confusion.mat, 
         pred.prob = list(train.prob=train.pred.res, test.prob=test.pred.res))

frag.train <- frag.res$pred.prob$train.prob
frag.test <- frag.res$pred.prob$test.prob
colnames(frag.train)[2:8] <- colnames(frag.train)[2:8]
colnames(frag.test)[2:8] <- colnames(frag.test)[2:8]
train <- frag.train[, c('sample_name', 'true.label', 'top1', 'top2', 'COREAD', 'ESCA', 'LIHC', 'Lung', 'OV', 'STAD', 'THCA')]
test <- frag.test[, c('sample_name', 'true.label', 'top1', 'top2', 'COREAD', 'ESCA', 'LIHC', 'Lung', 'OV', 'STAD', 'THCA')]
write.table(train, opt$fileouttrain, sep="\t", quote=F, row.names=F)
write.table(test, opt$fileouttest, sep="\t", quote=F, row.names=F)