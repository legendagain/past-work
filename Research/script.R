library(methods)
library(lightgbm)

# utilities
library(tidyverse)
library(binr)
library(text2vec)
library(tokenizers)
library(RecordLinkage)
library(dplyr)
library(magrittr)
library(caret)
library(Matrix)
library(parallel)
library(doParallel)

# model libraries
library(glmnetcr)
library(ordinalNet)
library(biglasso)
library(e1071)
library(klaR)
library(liquidSVM)

# initialization settings
memory.limit(size = 100000)
set.seed(42)
dev = TRUE
# cluster = makeCluster(detectCores() - 1)
# registerDoParallel(cluster)

# read in data
data = read_csv("Train_rev1.csv")

# replace missing values
# https://stackoverflow.com/questions/8161836/how-do-i-replace-na-values-with-zeros-in-an-r-dataframe
data = mutate_all(data, funs(replace(., is.na(.), "Missing")))  # Replace all missing values
gsubMany <- function(x, toReplace) {
  for (str in toReplace) {
    x = gsub(str, "", x)
    print(str)
  }
  return(x)
}

# todo:
# 1. ignore universities
# 2. manual checking list
data$Company = tolower(data$Company) %>%
  gsubMany(c("\\.", "@", "the", "of", "limited", "ltd", "plc", "llp", "group",
             "recruit", "recruitment", "recruiting", "consulting", "solutions", "solution", "services", "service",
             "associates", "healthcare", "education", "personnel", "resourcing")) %>%
  trimws
data$SalaryNormalized = as.numeric(data$SalaryNormalized)

# binning salary
data$SalaryBin = NA
binQty = 10
salaryBins = bins.greedy(data$SalaryNormalized, nbins = binQty)
for (i in seq(1, binQty)) {
  binlo = salaryBins$binlo[i]
  binhi = salaryBins$binhi[i]
  data[between(data$SalaryNormalized, binlo, binhi),]$SalaryBin = paste("X", binlo, ".", binhi, sep = "")
}
# make ordinal to allow ordinal logit regression
data$SalaryBin = as.ordered(data$SalaryBin)
data$SalaryLevels = as.numeric(data$SalaryBin)

# reduce sample size if in debug mode
if (dev) {
  data = data %>% group_by(SalaryLevels) %>% sample_frac(.05) %>% sample
  data = data[sample(nrow(data), nrow(data)),]  # randomize after grouping
}

# standardize Company names
company = unique(data$Company) %>%
  as.tibble()
colnames(company)[1] = "Company"

# de-duplicate Companies based on string dist
# https://stackoverflow.com/questions/39215184/generating-a-unique-id-column-for-large-dataset-with-the-recordlinkage-package
company %>%
  RLBigDataDedup(strcmp = TRUE, strcmpfun = "jarowinkler") %>%
  epiWeights() %>%
  epiClassify(0.92) %>%
  getPairs(filter.link = "link", single.rows = TRUE) -> matching_data

company.mapping = left_join(mutate(company, ID = 1:nrow(company)),
                   dplyr::select(matching_data, id.1, id.2) %>%
                     arrange(id.1) %>% filter(!duplicated(id.2)),
                   by = c("ID" = "id.2")) %>%
  mutate(ID = ifelse(is.na(id.1), ID, id.1)) %>%
  dplyr::select(-id.1)
colnames(company.mapping)[2] = "CompanyId"
data = left_join(data, company.mapping, by = "Company")
data$CompanyId2 = paste('C', data$CompanyId, sep="")

# somehow deduping results in missing Company IDs (169 records)
# so we're imputing it with the next highest value
if (sum(is.na(data$CompanyId) == TRUE) > 0) {
  data[!complete.cases(data$CompanyId),]$CompanyId = max(data$CompanyId, na.rm = TRUE) + 1
}

# sample before running any sort of model
sample = sample.int(n = nrow(data), size = floor(.6*nrow(data)), replace = FALSE)
train = data[sample, ]
test = data[-sample, ]

# bag of words dummifying
prep_fun = tolower
tok_fun = function(x) {
  word_tokenizer(x) %>% lapply(SnowballC::wordStem, language="en")
}

# Analysing Texts with the text2vec package
# https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html
createIterator <- function(data, ids) {
  iterator = itoken(data,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = ids,
                    progressbar = FALSE)
  return(iterator)
}

createDtm <- function(dt, ids) {
  iterator = createIterator(dt, ids)
  vocab = create_vocabulary(iterator, stopwords = stopwords()) %>%
    prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.75,
                     doc_proportion_min = 0.01, vocab_term_max = 10000)
  vectorizer = vocab_vectorizer(vocab)
  
  tfIdf = text2vec::TfIdf$new()
  dtm = create_dtm(iterator, vectorizer) %>%
    fit_transform(tfIdf)
  
  return(dtm)
}

# one-of-k encoding
# convert df to sparse matrix: https://stackoverflow.com/questions/27008633/r-data-frame-convert-to-sparse-matrix
createSparseMatrix <- function(dt) {
  dummies = sparse.model.matrix(~ -1 + ContractType + ContractTime + Category + SourceName, dt)
  #dummies = sparse.model.matrix(~ -1 + ContractType + ContractTime + CompanyId + Category + SourceName, dt)
  dtm = createDtm(paste(dt$Title, dt$FullDescription), dt$Id)
  vals = cBind(dummies, dtm)
  
  return(vals)
}

data.sparse = createSparseMatrix(data)
x.sparse = data.sparse[sample, ]
testX.sparse = data.sparse[-sample, ]
x = as.tibble(as.matrix(x.sparse))
testX = as.tibble(as.matrix(testX.sparse))
y = as.factor(make.names(train$SalaryBin))
testY = as.factor(make.names(test$SalaryBin))

# lightGBM
OptimLight <- function(x) {
  param = list(num_leaf = 31,
  #             learning_rate = 0.1,
  #             num_iterations = 50,
               objective = "multiclass",
               num_class = 10)
  lgbm.cv <- lgb.cv(param, data = x.sparse, label = as.numeric(y) - 1,
                    nfold = 5, eval = "multi_logloss")
}

lgbm <- lightgbm(data = x.sparse,
                 label = as.numeric(y) - 1,
                 num_leaves = 31,
                 #learning_rate = .1,
                 nrounds = 50,
                 objective = "multiclass",
                 #metric = "multi_error",
                 num_class = 10)
                 #categorical_feature = "CompanyId",
                 #tree_learner = "feature",
                 #boosting = "dart")
                 #max_bin = 63)
lgbmPred = predict(lgbm, testX.sparse, reshape = TRUE)
lgbmPred = apply(as.data.frame(lgbmPred), 1, which.max)
dist = mean(abs(lgbmPred - as.numeric(testY)))
error = 1 - mean(lgbmPred == as.numeric(testY))
print(dist)
print(error)

# hypertuning of LightGBM
grid_search = expand.grid(
  feature_fraction = c(1, .8, .2),
  num_leaves = c(31, 64, 127),
  learning_rate = c(.1, .2, .5, 1),
  max_depth = c(-1, 2, 63),
  scale_pos_weight = c(1, 10000),
  min_child_weight = c(.01, 7.343),
  nrounds = c(50,100,200)
)

#library(caTools)
#sample = sample.split(train, SplitRatio = .8)
#dtrain = lgb.Dataset(as.matrix(subset(x, sample == TRUE), sparse = TRUE),
#                     label = subset(as.numeric(y) - 1, sample == TRUE),
#                     free_raw_data = FALSE)
#dtest = lgb.Dataset(as.matrix(subset(x, sample == FALSE), sparse = TRUE),
#                    label = subset(as.numeric(y) - 1, sample == FALSE),
#                    free_raw_data = FALSE)
#valids<-list(test = dtest)

perf <- numeric(nrow(grid_search))
for (i in 1:1) {
  #model <- lgb.train(list(objective = "multiclass",
  #                        metric = "l2",
  #                        feature_fraction = grid_search[i, "feature_fraction"],
  #                        num_leaves = grid_search[i, "num_leaves"],
  #                        learning_rate = grid_search[i, "learning_rate"],
  #                        max_depth = grid_search[i, "max_depth"],
  #                        scale_pos_weight = grid_search[i, "scale_pos_weight"],
  #                        min_child_weight = grid_search[i, "grid_search_weight"]),
  #                   dtrain, nround = 100,
  #                   objective = "multiclass", num_class = 10,
  #                   valids = valids,
  #                   early_stopping_rounds = 10)
  lgbm = lightgbm(data = as.matrix(x),
                  max_bin = 100,
                  label = as.numeric(y) - 1,
                  num_leaves = grid_search[i,"num_leaves"],
                  learning_rate = grid_search[i,"learning_rate"],
                  nrounds = grid_search[i,"nrounds"],
                  max_depth = grid_search[i,"max_depth"],
                  scale_pos_weight = grid_search[i,"scale_pos_weight"],
                  min_child_weight = grid_search[i,"min_child_weight"],
                  objective = "multiclass",
                  num_class = 10)
  lgbmPred = predict(lgbm, testX.sparse, reshape = TRUE)
  lgbmPred = apply(as.data.frame(lgbmPred), 1, which.max)
  perf[i] = mean(abs(lgbmPred - as.numeric(testY)))
  gc(verbose = FALSE)
  #save.image("C:/R/job-kaggle/.RData")
}
cat("Model ", which.min(perf), " is lowest loss: ", min(perf), sep = "","\n")
print(grid_search[which.min(perf), ])

# 5-fold
folds = 5
cvIndex = createFolds(y, k = folds, returnTrain = FALSE)
tc = trainControl("cv", index = cvIndex, number = folds, classProbs = TRUE,
                  summaryFunction = multiClassSummary, allowParallel = FALSE,
                  returnData = FALSE, trim = TRUE)

# tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(10,-2, length = 10))
tuneGrid = expand.grid(alpha = 1, lambda = .01)
glmnet.fit3 = train(x.sparse, y,
                   method = "glmnet",
                   trControl = tc,
                   family = "multinomial",
                   tuneGrid = tuneGrid)

# ordinalNet
mnlogit = list(type = "Classification",
               library = "mnlogit",
               loop = NULL)
# xgboost
glmnet.fit = train(x.sparse, y,
                   method = "glmnet",
                   trControl = tc,
                   family = "multinomial")

# https://cran.r-project.org/web/packages/glmnetcr/vignettes/glmnetcr.pdf
fit = cv.glmnet(x.sparse, y, nfolds = 5, family = "multinomial", type.measure = "class")
# fit = glmnetcr(x, y, method = "forward")
# bestlam = min(fit$lambda)
bestlam = fit$lambda.1se
pred = predict(fit, s = bestlam, newx = testX.sparse, type="class")
dist = mean(abs(as.numeric(as.factor(lgbmPred)) - as.numeric(testY)))

# SVMLight - slow!
svmlPath = "C:\\svm_light_windows64"
svml.fit = svmlight(x = x, grouping = y,
                    pathsvm = svmlPath,
                    #temp.dir = paste(svmlPath, "\temp", sep=""),
                    out = TRUE,
                    svm.options = "-n 8 -# 1000 -m 1024 -v 3")
svmlPred = predict(svml.fit, testX)

# LiquidSVM
lsvm.fit = mcSVM(x, y, mc_type="AvA_ls")
lsvmPred = predict(lsvm.fit, testX)
dist = mean(abs(as.numeric(lsvmPred) - as.numeric(testY)))

# https://www.r-bloggers.com/machine-learning-using-support-vector-machines/
fit.svm = svm(y ~ x, x = x, y = y, type = "C-classification")

# reporting error
# https://stackoverflow.com/questions/36121171/how-to-extract-actual-classification-error-rate-with-cost-function-from-cv-glmne
error = mean(pred != testY)

# function to visualize sparse matrices
sparse2triples <- function(m) {
  SM = summary(m)
  D1 = m@Dimnames[[1]][SM[,1]]
  D2 = m@Dimnames[[2]][SM[,2]]
  data.frame(row=D1, col=D2, x=m@x)
}

# other matrix utility functions
sparse.matrix.write = function (sm, filename = "matrix-output.csv", nrow = 100) {
  m = as.matrix(sm[1:nrow,])
  matrix.write(m, filename)
}

matrix.write = function (m, filename = "matrix-output.csv") {
  write.table(m, file = filename, row.names = TRUE, col.names = TRUE, sep = ",")
}

grid_search = grid_search[order(perf)[seq(1,150,3)],]