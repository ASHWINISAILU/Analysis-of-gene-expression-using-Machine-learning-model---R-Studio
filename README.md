Analysis of gene expression using Machine learning models - R Studio

**Abstract:**
The main aim of this project is to analyze gene expression on invasive vs non-invasive using machine
learning algorithms in R language. The report summarizes the initial data preprocessing and analysis
of the different models of both supervised and unsupervised models. And find the best machine
learning model to analyze the gene expression of invasive and non-invasive cancer. We have
implemented a new model to try to improvise our best-supervised machine learning model using an
unsupervised machine learning model (k clustering).


**Introduction:**
Our initial task is to analysis on the data set and identify the data type, distribution of data, and analysis of various variables. As the data set is huge, we have implemented PCA for unsupervised learning models and t-test for supervised learning models. A list of machine learning models of both supervised and unsupervised models was implemented. Using resampling techniques, the different machine learning models were compared and the best machine learning was identified. On our research question, we tried to improvise the clustering using our best machine learning model.

**Analysis**


**Descriptive and graphical analysis**

We are analyzing numerical data representing gene expressions in both invasive and non-invasive cancer genes. Initially, the dataset consists of 4949 variables indicating different genes among 78 patients, identified using the dim() function. We utilized the largest registered number within our team (2312181) as a seed to generate uniform random numbers from the provided CSV file. Subsequently, we created a subset consisting of 4948 randomly selected and ranked numbers, from which we extracted the first 2000 as our subset, referred to as 'team_gene_subset'. Upon inspection, we discovered 57 missing values within the dataset, which we addressed by replacing them with the median of non-missing values in each respective column. To manage outliers within the dataset, we employed the Winsorizer method. Initially, around 1602 variables exhibited outliers, which we successfully reduced to 245, thereby improving the dataset by minimizing the presence of outliers.

![image](https://github.com/user-attachments/assets/552f64b2-a224-4bb0-a147-4a3b08095dd0)


**Dimensional Reduction**

Dimension reduction is a distinct technique used in data preprocessing to reduce the dimensions of the dataset. This helps to reduce the number of features while retaining the most important information. We have used Principal component analysis to reduce dimensions for unsupervised learning models and a two-sample t-test to reduce dimensions for supervised learning models.

**PCA**
PCA is used as dimensional reduction here as it retains the maximum originality of the data and maximum variance in the first few components, reducing the noise by focusing on the direction of maximum variance. PCA reduces dimensions by selecting crucial PCs, retaining much original data. The high cumulative variance explained suggests effective dimensionality reduction. Initial principal components PC1, PC2, and PC3 show higher variability and explain more variance than subsequent ones.


**t-Test**
A sample t-test is performed comparing the gene expression levels between two classes or groups represented by the "Class" variable. Genes with significant differences in expression levels between invasive and non-invasive cancer (class) are retained, reducing the dataset's dimensionality.


**Unsupervised learning model**

Machine learning models like PCA, k-means clustering, hierarchical clustering, and additionally t- SNE (t-distributed stochastic neighbor embedding) have been trained using the dimensional reduced dataset. Below are the inferences of each model:

**PCA:**

The significance of principal components (PCs) is determined by their ranking, where PC1, having the highest standard deviation and explaining the most variance, holds the most significance, followed by PC2, PC3, and so forth. PC1's standard deviation of 1.41421 exceeds that of subsequent components, signifying its capture of the greatest variability in the dataset. Each PC's proportion of variance reveals its contribution to the overall dataset variance, with PC1 explaining 2.564% followed by PC2 with 1.282%, and so on. The cumulative proportion of variance demonstrates the combined impact of each PC, aiding in understanding information preservation as more PCs are considered. PC1 through PC22 collectively explain a significant portion of the dataset's variance, indicating their capture of fundamental patterns. These components are crucial for dimensionality reduction and feature selection, offering substantial information retention while reducing dimensionality. The PCA model supports applications like dimensionality reduction, visualization, and feature selection, aiding in identifying significant genes or features. Researchers can determine the optimal number of retained principal components based on desired information retention levels. PC78, with a near-zero standard deviation and negligible variance, holds minimal relevance and can be disregarded in further analyses. In conclusion, PCA analysis provides valuable insights into dataset structure, guiding future investigations related to invasive and non-invasive cancer genes.


**k-mean clustering**


We see distinct two clusters formed. The optimal k value is 2, identified using the Silhouette method. This method provides the optimal score of the k value, and considers both cohesion and separation, providing a comprehensive measure of cluster quality rather than other methods focus on cluster variances. We see two distinct clusters formed with k=2.


![image](https://github.com/user-attachments/assets/258e0091-5bcb-4ad3-89cc-f063a8de67cd)


![image](https://github.com/user-attachments/assets/446a6ddb-b786-483e-b291-63bccd506be8)


**Hierarchical clustering**

The hierarchical clustering employed the complete linkage method and Euclidean distance to analyze 2000 genes representing patients. This method forms clusters based on maximum inter-cluster distance, yielding compact clusters. It organizes data into a hierarchical structure, grouping similar patients. In summary, hierarchical clustering offers a structured method for uncovering the dataset's inherent organization. It enables the recognition of specific patient clusters sharing similar gene expression patterns linked to both invasive and noninvasive cancer genes.



**t-SNE(additional model)**

We have an additional unsupervised machine learning model t-distributed stochastic neighbor embedding to visualize high dimensional data given each data point with a location. We have set the parameters PCA as a matrix excluding the last column. The t-SNE analysis utilized OpenMP with 1 thread, 2 dimensions, perplexity 10, and theta 0.5. It computed similarities, built a tree, and learned to embed. Error decreased to 0.231252 over 1000 iterations. Fitting took 0.13 seconds. The below visualization shows the dimensions of two clusters using t-SNE model.


![image](https://github.com/user-attachments/assets/55a548d3-e2f9-403d-8ab7-0908ae541ee3)


**Supervised learning model **

Supervised models are trained and predicted using the reduced dimensional dataset from a two-sample t-test. We have set the seed to ensure a random process, with the class label "Y" and predictor label "X" assigned to the matrix labelled "signi_gene_matrix" which contains the features related to the classification task. The data is split into training and testing as an 80:20 ratio ensuring model performance is evaluated on unseen data. 3-fold cross-validation is performed dividing the data into 3 subsets for training and validation. We have set accuracy as the default e-evaluation metric. The models consider 64 samples and 343 predictors to predict the classes. Sample size ranges from 42,43,43 respectively.

**Logistic Regression **

The model achieved an accuracy rate of approximately 51.52%. Regarding agreement between observed and predicted classes, the Kappa statistic measured at 0.004602. The F1 score, which balances precision and recall, stood at 0.416122. Sensitivity, indicating the proportion of actual positives correctly identified, was 39.26%, while specificity, representing the proportion of actual negatives correctly identified, reached 61.11%. The precision for positive cases was 44.44%, with a negative predictive value of 56.23%. The detection rate, or the proportion of actual positives correctly classified, was 17.17%. Considering class imbalance, the balanced accuracy was calculated at 50.19%. Overall, the logistic regression model shows potential for improvement, as its accuracy slightly surpasses random guessing.


**LDA(Linear Discriminant Analysis)**

The accuracy of the LDA model without implementing any preprocessing technique is 78.07%. Kappa is 0.5642 agreement between the predicted and actual class which is a moderate measure of agreement that accounts for the agreement occurring by chance alone. The F1 score is 0.7652174 suggests balanced precision and recall, indicating effective classification. Specificity (0.778) identifies noninvasive cancer genes accurately, aiding true negative identification. A positive Predictive Value (0.7904) shows the Likelihood of correctly identifying invasive cancer genes among positive predictions. Balanced Accuracy (0.7815) offers balanced classification accuracy considering class imbalances. The LDA model shows potential in distinguishing invasive and noninvasive cancer genes, with decent accuracy and balanced metrics.


**QDA (Quadratic Discriminant Analysis)**


Our QDA model was unfit for our random dataset as the model was unable to run a small group. The error pops up when the covariance matrices for each class are calculated.

 
**k-NN(k- Nearest Neighbors)**


The model with k = 5 was chosen as optimal due to its highest accuracy. The achieved accuracy value is 71.86%, which indicates the proportion of correctly classified cases. Kappa indicates moderate agreement between predicted and actual classification. The F1 score shows a good balance between precision and recall. The sensitivity (recall) is approximately 72.22%, indicating the ability to correctly detect invasive cancer genes. The accuracy (Pos_Pred_Value) is approximately 71.46%, which reflects the proportion of true invasive cancer gene predictions among all positive predictions. Neg_Pred_Value is 78.84. % indicating the proportion of true non-invasive cancer predictions out of all negative predictions. The average sensitivity and specificity reaches about 73.61%.
Random Forest The random forest model achieved an accuracy of 75.04 with mtry=2, effectively classifying invasive and noninvasive cancer genes. However, for mtry values of 172 and 343, accuracy dropped to 0.64. Despite demonstrating reasonable sensitivity and specificity, additional tuning and validation can support better performance.

**Random Forest**

The random forest model achieved an accuracy of 75.04 with mtry=2, effectively classifying invasive and noninvasive cancer genes. However, for mtry values of 172 and 343, accuracy dropped to 0.64. Despite demonstrating reasonable sensitivity and specificity, additional tuning and validation can support better performance.


**SVM (Support Vector Machine)**

The SVM model achieved an accuracy of 73.3%, indicating its proficiency in classifying invasive and non-invasive cancer genes. With a kappa value of 0.522, the model exhibits moderate agreement beyond chance. An F1 score of 0.736 suggests a balance between precision and recall, enhancing its overall performance. The model demonstrates a sensitivity of 0.778 and a specificity of 0.750, indicating its ability to accurately identify both types of cancer genes. Positive predictive value (PPV) and negative predictive value (NPV) stand at 0.700 and 0.826, respectively, reflecting the model's predictive capability for positive and negative cases. A detection rate of 0.342 indicates the model's effectiveness in identifying positive cases relative to the total instances. With a balanced accuracy of 0.764, the model provides a fair estimate considering class distribution imbalance. The tuning parameter "C" remained constant at 1, suggesting consistent model performance across various regularization strengths.


**GBM- (additional model)**

The GBM model underwent evaluation with various tuning parameters such as interaction depth and the number of trees. Performance metrics including accuracy, Kappa statistic, F1 score, sensitivity, specificity, positive predictive value, and negative predictive value were analyzed across different parameter combinations. The optimal model, chosen based on accuracy, maintained a shrinkage parameter at 0.1 and a minimum of 10 observations in each terminal node. Accuracy ranged from approximately 69.05% to 83.33%, with Kappa statistics varying from 34.85% to 66.17%, and F1 scores from 58.27% to 80.99%. The optimal configuration featured an interaction depth of 3, 100 trees, shrinkage of 0.1, and a minimum of 10 observations in each terminal node. Overall, the GBM model shows promise in predicting invasive and noninvasive cancer genes.


**Resampling technique – Cross validation**

k-fold cross-validation techniques are used as a resampling technique in this dataset. It trains the unseen data by shuffling the data randomly and setting k groups. Each group acts as test data while the remaining groups act as training data. Later fit a model on the training set and evaluate it on the test set. Generally, we started k=10 to train the data however we could see the accuracy of all supervised models tend to increase as the k value decreased. We could k=3 gave the maximum accuracy of the models. This finding suggests that with a lower number of folds in the cross-validation process, the model tends to generalize better to unseen data.In other words, by reducing the number of folds, the model captures more robust patterns in the data, resulting in higher accuracy.



**Best model and implementation of k cluster to improve best model**


The best model identified as GBM (Gradient Boosting Machine) which shows 83.33% while using the 3-fold cross-validation technique. GBM, short for Gradient Boosting Machine, is a potent algorithm in machine learning recognized for its exceptional predictive precision and resilience. Within the provided table, GBM showcases impressive performance across diverse evaluation criteria. It attains an average accuracy of around 83.33%, positioning it as one of the leading models in the comparison. Furthermore, GBM demonstrates elevated values for several other metrics including Balanced Accuracy, F1 Score, Kappa, Sensitivity, and Specificity, underscoring its proficiency in precisely distinguishing between invasive and non-invasive genes. These findings underscore GBM's potential as a promising option for predictive modeling endeavors in gene expression analysis.

As per our research question, we have implemented our k cluster model to check if our GBM model is improvising. The model achieved an accuracy rate of 85.71%, suggesting that it accurately classified around 85.71% of instances into their respective categories (invasive or non-invasive cancer). With a Kappa statistic of 0.6957, there is substantial agreement between the model's predictions and the actual classes, surpassing what would be expected by chance alone. Sensitivity, at 66.67%, indicates the model's ability to correctly identify 66.67% of true positive cases (invasive cancer) among all actual positive cases. Specificity is at 100%, meaning the model accurately identifies all true negative cases (non-invasive cancer) among all actual negative cases. A positive predictive value of 1 implies that the model correctly predicts instances as positive (invasive cancer) 100% of the time. Conversely, a negative predictive value of 0.8 indicates that the model correctly predicts instances as negative (non-invasive cancer) 80% of the time. The dataset exhibits a prevalence of invasive cancer at approximately 42.86%. The model successfully detects around 28.57% of all invasive cancer cases.


![image](https://github.com/user-attachments/assets/402b2850-793e-4181-8ad5-90535c084014)


![image](https://github.com/user-attachments/assets/8c5b256d-bc01-48c5-8493-639dc9eb30c9)
![image](https://github.com/user-attachments/assets/26dc82e0-2d1e-4d33-a827-901e857fbbec)
![image](https://github.com/user-attachments/assets/1844406b-6870-4e0c-ac55-2937ecfb97e0)
![image](https://github.com/user-attachments/assets/4d401132-13b5-42ba-9b6c-36c142afe94f)



**Conclusion**


Invasive and non-invasive cancer genes have different gene expression patterns, which are crucial for understanding cancer progression. This report analyses the predictions of the Gradient Boosting Machine (GBM) model to classify gene expression as invasive or non-invasive with accuracy of 85.71% which has improved to 2.38% from the initial GBM model.
In summary, the GBM model, when trained on gene expression data alongside supplementary cluster features, shows encouraging capabilities in discerning between invasive and non-invasive cancer types. The model showcases notable accuracy, significant concordance with real classes, and favourable sensitivity and specificity.


**References:**
•
An Introduction to Statistical Learning with Applications in R- Gareth ,James Daniela Witten,Trevor Hastie ,Robert Tulshiram
•
https://www.datacamp.com/tutorial/pca-analysis-r
•
https://www.datacamp.com/tutorial/k-means-clustering-r
•
https://www.datacamp.com/tutorial/machine-learning-in-r




