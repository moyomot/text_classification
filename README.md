# Text Classification
Text classification is an important theme and basic classification task in natural language processing.  
There are many methods to classify texts. Recently classification method by deep learning has been invented.  
I will introduce classification by CNN and LSTM which is a representative deep learning.  
And also summarized svm and naive bayes which are classic methods.

## Experiments
### Classifiers
Classifier | Paper Link
-- | -- 
CNN | [Paper](http://www.aclweb.org/anthology/D14-1181)
LSTM | -
Character Level CNN | [Paper](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
SVM | [scikit-earn](http://scikit-learn.org/stable/modules/svm.html#svm-classification)
Naive Bayes | [scikit-learn](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)


### Result
#### AG's News
Classes | Train Data Size | Test DataSize
-- | -- | --
4 | 120,000 | 7,600

Classifier | validation loss | validation accuracy
-- | -- | --
CNN | 0.2994 | 0.9055
LSTM | 0.2587 | 0.9106
Character Level CNN | 0.3692 | 0.8709
SVM | - | 0.9007
Naive Bayes | - | 0.9182
