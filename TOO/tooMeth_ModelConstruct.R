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
  make_option("--fileintrain",  action = "store"),
  make_option("--fileintest",  action = "store"),
  make_option("--feature",  action = "store"),
  make_option("--sampleinfo",  action = "store"),
  make_option("--fileouttrain",  action = "store"),
  make_option("--fileouttest",  action = "store"),
  make_option("--fileoutfig",  action = "store")
)
opt = parse_args(OptionParser(option_list = option_list))

train.imputed <- fread(opt$fileintrain, header = T) %>% 
  as.data.frame()

test.imputed <- fread(opt$fileintest, header = T) %>% 
  as.data.frame()
# train/test.imputed format
# sample_name    sample_type    gender    feature1    feature2    ...    featurem
# sample1    Colorectum    Male    1    0    ...    0.858160068356593

boruta.confirm <- fread(opt$feature, header = T) %>%
  as.data.frame()
# boruta.confirm format
# type    marker_index    meanImp    medianImp    minImp    maxImp    normHits    decision
# lung    marker1    7.30117446850714    7.50072398794453    2.87728192057872    9.33976359629354    1    Confirmed

# Read sample information
info <- read.table(opt$sampleinfo, header=T, sep="\t") %>%
  dplyr::select(c(sample_name, sample_type, cohort, gender)) %>% 
  dplyr::filter(sample_type!="Healthy")

# 替换为正确的性别
train.imputed$gender <- info$gender[match(train.imputed$`sample_name`, info$`sample_name`)]
test.imputed$gender <- info$gender[match(test.imputed$`sample_name`, info$`sample_name`)]

confirm.top <- boruta.confirm %>%
  group_by(type) %>%
  slice_head(n = 30)

marker <- confirm.top$marker_index %>% unique()


################################ 特征筛选 ###################################
# step3: Train the model
train <- train.imputed %>% dplyr::select(sample_name, sample_type, gender, all_of(marker))
test <- test.imputed %>% dplyr::select(sample_name, sample_type, gender, all_of(marker))

control <- trainControl(method = "cv", number = 5, 
                        classProbs = TRUE, # search = "random",
                        summaryFunction = multiClassSummary, allowParallel = T)

hyper_grid <- expand.grid(
  alpha = seq(0.1, 1, by = 0.1),
  lambda = seq(0.1, 1, by = 0.1))

# Grid search using the train function
set.seed(123)
cl <- makePSOCKcluster(20)
registerDoParallel(cl)

m_glmnet <- train(
  sample_type ~ ., data = train[, -c(1)], method = "glmnet", metric = "Kappa",
  trControl = control,
  tuneGrid = hyper_grid)

stopCluster(cl)

best_lambda <- m_glmnet$bestTune$lambda
best_alpha <- m_glmnet$bestTune$alpha
cofficient <- coef(m_glmnet$finalModel, s = best_lambda)

# step4: Evaluation on the Test Set
# Predictions
# Lung, Colorectum, Stomach, Liver, Esophagus, Thyroid, Ovary

raw.pred.train <- predict(m_glmnet, train, type = "raw")

raw.pred.train <- factor(raw.pred.train, levels = c("Lung", "Colorectum", "Stomach",
                                                  "Liver", "Esophagus", "Thyroid",
                                                  "Ovary"))

actual.label <- factor(train$sample_type, levels = c("Lung", "Colorectum", "Stomach",
                                                  "Liver", "Esophagus", "Thyroid",
                                                  "Ovary"))

mat.train <- confusionMatrix(raw.pred.train, actual.label)
print(mat.train)

raw.pred.test <- predict(m_glmnet, test, type = "raw")
raw.pred.test <- factor(raw.pred.test, levels = c("Lung", "Colorectum", "Stomach",
                                                  "Liver", "Esophagus", "Thyroid",
                                                  "Ovary"))
actual.label <- factor(test$sample_type, levels = c("Lung", "Colorectum", "Stomach",
                                                  "Liver", "Esophagus", "Thyroid",
                                                  "Ovary"))

mat.test <- confusionMatrix(raw.pred.test, actual.label)
print(mat.test)

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

confusion.mat <- list(train.mat = mat.train$table, test.mat = mat.test$table)

uraModel.res <- list(model=m_glmnet, confirmed.var=var, acc=acc, confusion.mat=confusion.mat, 
                     pred.prob = list(train.prob=train.pred.res, 
                                      test.prob=test.pred.res))
ura.train <- uraModel.res$pred.prob$train.prob
ura.test <- uraModel.res$pred.prob$test.prob
colnames(ura.train)[2:8] <- colnames(ura.train)[2:8]
colnames(ura.test)[2:8] <- colnames(ura.test)[2:8]
train <- ura.train[, c('sample_name', 'true.label', 'top1', 'top2', 'Colorectum', 'Esophagus', 'Liver', 'Lung', 'Ovary', 'Stomach', 'Thyroid')]
test <- ura.test[, c('sample_name', 'true.label', 'top1', 'top2', 'Colorectum', 'Esophagus', 'Liver', 'Lung', 'Ovary', 'Stomach', 'Thyroid')]
write.table(train, opt$fileouttrain, sep="\t", quote=F, row.names=F)
write.table(test, opt$fileouttest, sep="\t", quote=F, row.names=F)

save(uraModel.res, file = "res.rdata")
# 混淆矩阵热图
mat.train <- uraModel.res$confusion.mat$train.mat
mat.test <- uraModel.res$confusion.mat$test.mat 

train_numbers <- matrix(sprintf("%d", mat.train), nrow = nrow(mat.train))

test_numbers <- matrix(sprintf("%d", mat.test), nrow = nrow(mat.test))

# 将矩阵转换为长格式的数据框
# 将矩阵转换为长格式的数据框
data1 <- melt(mat.train)
numbers1 <- melt(train_numbers)  
data2 <- melt(mat.test)
numbers2 <- melt(test_numbers)

pdf(opt$fileoutfig)
# 绘制第一个热图
ggplot(data1, aes(Reference, Prediction, fill = value)) + 
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "#F39B7FFF") +
  geom_text(aes(label = numbers1$value), color = "black", size = 6) +  # 添加显示数字
  theme_minimal() +
  labs(x = "True label", y = "Predicted label") +  # 添加标题和坐标轴标签
  theme(
    axis.text.x = element_text(size = 16, angle = 90, color = "black"),
    axis.text.y = element_text(size = 16, color = "black"),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(hjust = 0.5, size = 18),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.key.size = unit(1.2, "cm")  # 调大颜色条的尺寸
  )
# 绘制第二个热图
ggplot(data2, aes(Reference, Prediction, fill = value)) + 
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "#F39B7FFF") +
  geom_text(aes(label = numbers2$value), color = "black", size = 6) +  # 添加显示数字
  theme_minimal() +
  labs(x = "True label", y = "Predicted label") +  # 添加标题和坐标轴标签
  theme(
    axis.text.x = element_text(size = 16, angle = 90, color = "black"),
    axis.text.y = element_text(size = 16, color = "black"),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(hjust = 0.5, size = 18),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.key.size = unit(1.2, "cm")  # 调大颜色条的尺寸
  )
dev.off()