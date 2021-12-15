# Sentiment Analysis of Yelp Hotel Reviews
      
Mary Kryslette C. Bunyi   
PPOL566 (Data Science III) Extra Credit Project

# Background
**Research Question:** How well do unlabeled sentiment analysis models approximate the actual star ratings that reviewers give? Can we do away with the star rating requirement and still gain accurate insight into customers' sentiment? What are the tradeoffs, if any? 

**Objective:** Compare the unlabeled approaches -- lexical-based (TextBlob), transfer learning/DL (transformers) -- to the ground truth labels/ratings supplied by the customers and determine how well the alternative approaches perform

# Methods
Exploratory data analysis shows that the dataset is largely imbalanced and skewed towards higher ratings (i.e., 4 and 5). Mapping the 1-5 Yelp rating scale to positive (4-5) and negative (1-3) sentiment still shows an imbalance. However, this can be remedied by taking equal-sized samples for our sentiment analysis models.
![image.png](attachment:image.png)

![image.png](attachment:image.png)

To speed up computation, I took a sample of 500 observations each from the set of positive and negative reviews. I also cleaned the text by expanding contractions (which may be important since they may denote negation) and removing non-letters.

I ran the lexicon-based (TextBlob) and transfer learning (BERT sentiment analysis) models to both the raw and cleaned data to check for the robustness of their performances. Since TextBlob outputs a polarity score instead of binary labels, I determined the cut-off scores based on the value where the distributions of the true positive and negative sentiments meet. Above this cut-off, reviews are labeled positive. Below this cut-off, reviews are identified as negative.

![image.png](attachment:image.png)

![image.png](attachment:image.png)

For the transfer learning approach, I used the default BERT sentiment analysis model. However, its implementation cannot handle text with more than 512 tokens. Thus, I set "truncation" to True, which means that for very long reviews, the classifier considers only the first 512 tokens. I do not expect this to have a significant effect on the results.

# Findings

Both the TextBlob and the BERT models registered above-70% accuracy, precision, recall, and F1 scores for both raw and cleaned data. These are good results considering that we are using unsupervised learning approaches. We could expect performance to further improve when the models train on the actual Yelp dataset.

A key objective of sentiment analysis on customer reviews is the improvement of products and services. While the transfer learning model edges the lexicon-based model across all 4 evaluation metrics chosen, if we would like to capture as much of the negative reviews as possible, applying TextBlob on Raw Data may be the best choice as it yields the highest number of true negatives (i.e., correctly tagged negative reviews) and the smallest number of false positives (i.e., negative reviews that the model mistook as positive). However, the trade-off for this model would be the presence of higher false negatives.

![image.png](attachment:image.png)

![image.png](attachment:image.png)

Word clouds of misclassified reviews (excluding the usual stopwords) show that there is considerable similarity between misclassified positive and negative reviews. For instance, misclassifications of the TextBlob model on raw data similarly mentioned words like good, place, like, great, and food. Aside from contributing to a whole thought/idea, many of these words express sentiment. It may be challenging to consider these as stopwords. Doing so may disrupt the meaning and sentiment of text. (The appendix contains word clouds for the other models, which generated similar results.)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

# Conclusions
Sentiment analysis models can provide a good approximation of customer sentiment. Both the lexicon-based (TextBlob) and transfer learning-based (BERT) sentiment analysis models performed well in predicting reviewers' actual ratings. Both registered above-70% scores across all evaluation metrics (i.e., accuracy, precision, recall, and F1 scores). 

While transfer learning is superior to the lexicon-based approach across all evaluation metrics, if our key objective is to identify areas for improvement, then it may be worthwhile to consider the model that captures the widest range of negative reviews possible. This is the TextBlob model as applied to raw review content. This model generates the most true negatives and the fewest false positives. However, given its propensity to identify reviews as negative, the trade-off for using this model would be the presence of higher false negatives.

Performance would only improve once the models train on the actual Yelp dataset. Nonetheless, these models are only able to distinguish between positive and negative sentiments, whereas the star ratings are more granular and add a dimension of intensity to customer sentiment. The difference in intensity may be valuable especially as extremely negative or positive ratings will tend to sway prospective customers. At the same time, "1" rating reviews may highlight the most pressing areas of concern while "5" rating reviews may surface comparative advantages that the hotels can best capitalize. In this regard, the 1-5 star rating scale can still be valuable.

# Recommendations
Given our large corpus of Yelp reviews which contains labeled data, we could easily train our model on the existing dataset. We could also upgrade the current models to perform multi-class predictions. For instance, TextBlob polarity scores could be mapped to discrete ratings. Instead of the default transformer model for sentiment analysis, we could also try using more focused pretrained models such as "nlptown/bert-base-multilingual-uncased-sentiment", which was finetuned specifically on product reviews and is able to predict 1 to 5 ratings.

We could likewise explore using our models in conjunction with other transformer models such as text summarization and named entity recognition. First, text summarization can help in extracting key information especially from longer reviews which are beyond the parsing capacity of the transfer learning (BERT) sentiment analysis model. Text summarization may also help in removing noise and keeping only the salient points of a review, thereby increasing label accuracy. Second, named entity recognition can help in reducing noise by dropping proper nouns, which do not express any sentiment.

As mentioned, while the sentiment analysis models performed well in predicting the reviewers' overall sentiment, the 1-5 star rating scale can still be valuable as they add a layer of intensity to the sentiment. To keep the it from becoming a deterrent, the discrete rating could be turned optional.

Similarly, it is possible that the sentiments of a significant proportion of customers remain unrecorded because they do not want to type text reviews. In this case, to capture more accurate ratings for the hotels, the text review could become optional as well.

Customers may also be given the option to provide more granular discrete ratings across different aspects of the experience. These ratings could be compared against text reviews of customers who choose not to select discrete ratings. More granular ratings may likewise show specific areas (e.g., service, food) where the companies can focus, which the text ratings may then provide further detail into.

# Annex: Code

# Exploratory Data Analysis

The dataset is hugely imbalanced. This can be remedied by taking equal-sized samples for our sentiment analysis models.

# TextBlob (lexicon-based)

# Transformers (Transfer Learning)

The BERT implementation cannot handle text with more than 512 tokens. Thus, we set truncation to True and take only the first 512 tokens, cutting off extremely long reviews. This should not have a significant effect on our results.

# Model evaluation

We want to capture as much of the negative reviews as possible. TextBlob on Raw Data has the highest number of true negatives and the least false positives. However, this comes at the expense of more false negatives.

We get evaluation metric scores higher than 70% across the board. Since we used unsupervised learning models, i.e.: did not train on the Yelp review data, these scores may indicate good performance.
