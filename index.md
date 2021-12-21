### Tableau Dashboard

<iframe seamless frameborder="0" src="https://public.tableau.com/views/1FinancialResiliencevsFinancialAccess/Dashboard1?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/1FinancialResiliencevsFinancialAccess/Dashboard1?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/1FinancialResiliencevsFinancialAccess/Dashboard2?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/2FinancialResiliencevsIncomeInequality/Dashboard1?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/access?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/inclusion?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/inclusion_poorest?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/borrow?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/save?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/reason?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '600' scrolling='yes' ></iframe>




## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/mkbunyi/Data-Viz-Final-Project/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mkbunyi/Data-Viz-Final-Project/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

### Plotly Viz Here
chart-studio.plot.ly
Upload to plotly
Embed iframe code
 



# Sentiment Analysis of Yelp Hotel Reviews

## Run in Anaconda:
jupyter nbconvert index.ipynb --to markdown --TemplateExporter.exclude_input=True
jupyter nbconvert index.ipynb --to markdown --TemplateExporter.exclude_code_cell=True
## 
   
Mary Kryslette C. Bunyi   
PPOL566 (Data Science III) Extra Credit Project

# Background
**Research Question:** How well do unlabeled sentiment analysis models approximate the actual star ratings that reviewers give? Can we do away with the star rating requirement and still gain accurate insight into customers' sentiment? What are the tradeoffs, if any? 

**Objective:** Compare the unlabeled approaches -- lexical-based (TextBlob), transfer learning/DL (transformers) -- to the ground truth labels/ratings supplied by the customers and determine how well the alternative approaches perform

# Methods
Exploratory data analysis shows that the dataset is largely imbalanced and skewed towards higher ratings (i.e., 4 and 5). Mapping the 1-5 Yelp rating scale to positive (4-5) and negative (1-3) sentiment still shows an imbalance. However, this can be remedied by taking equal-sized samples for our sentiment analysis models.


<img align="left" src="index_files/index_72_0.png" width="600px">
To speed up computation, I took a sample of 500 observations each from the set of positive and negative reviews. I also cleaned the text by expanding contractions (which may be important since they may denote negation) and removing non-letters.

<br clear="left">
<br clear="left">
<br clear="left">

<img align="left" src="index_files/index_67_0.png" height="200px">
<img align="right" src="index_files/index_68_0.png" height="200px">

<br clear="left">
<br clear="left">

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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>reviewID</th>
      <th>reviewerID</th>
      <th>reviewContent</th>
      <th>rating</th>
      <th>usefulCount</th>
      <th>coolCount</th>
      <th>funnyCount</th>
      <th>flagged</th>
      <th>hotelID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6/8/2011</td>
      <td>MyNjnxzZVTPq</td>
      <td>IFTr6_6NI4CgCVavIL9k5g</td>
      <td>Let me begin by saying that there are two kinds of people, those who will give the Tokyo Hotel 5 stars and rave about it to everyone they know, or... people who can't get past the broken phone, blood stains, beeping fire alarms, peg-legged receptionist, lack of water pressure, cracked walls, strange smells, questionable elevator, televisions left to die after the digital conversion, and the possibility that the air conditioner may fall out the window at any moment. That being said, I whole-heartedly give the Tokyo Hotel 5 stars. This is not a place to quietly slip in and out of with nothing to show but a faint memory of the imitation Thomas Kinkade painting bolted to the wall above your bed. And, there is no continental breakfast or coffee in the lobby. There are a few vending machines, but I'm pretty sure they wont take change minted after 1970. Here your senses will be assaulted, and after you leave you will have enough memories to compete with a 1,000 mile road-trip. I beg anyone who is even mildly considering staying here to give it a chance. The location is prime. We were able to walk down Michigan Ave and the river-walk in the middle of the night, all without straying too far from the hotel. There is a grocery store a block away and parking (which may cost more that your hotel room) across the street. Besides, this place is cheap. Super-cheap for downtown Chicago. The closest price we found in the area was four times as expensive. But, be sure to grab some cash. They don't accept credit cards. Some rules though: - Say hello to Clifton Jackson, the homeless guy by Jewel-Osco. - Buy him a drink, some chicken and look him up on Facebook. - Stay on the 17 floor. All the way at the top. - Go out the fire escape (be sure to prop the door open or you'll have a looong walk down) - Be very very careful. - Explore. (Yes, that ladder will hold your weight) - Be very very careful. - Don't be alarmed by any weird noises you hear. - Spend the night on the roof. 17 stories up, in the heart of Chicago. - Write your own Yelp review. I want to see that others are getting the Tokyo Hotel Experience. - Check out is at noon. Be sure to drink lots of water. - Spend the next day hung over. And... Please be careful on the roof.</td>
      <td>5</td>
      <td>18</td>
      <td>11</td>
      <td>28</td>
      <td>N</td>
      <td>tQfLGoolUMu2J0igcWcoZg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8/30/2011</td>
      <td>BdD7fsPqHQL73hwENEDT-Q</td>
      <td>c_-hF15XgNhlyy_TqzmdaA</td>
      <td>The only place inside the Loop that you can stay for $55/night. Also, the only place you can have a picnic dinner and get a little frisky on the 17th floor roof and then wake up in your room the next morning to an army of ants going in on your picnic leftovers.</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>N</td>
      <td>tQfLGoolUMu2J0igcWcoZg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6/26/2009</td>
      <td>BfhqiyfC</td>
      <td>CiwZ6S5ZizAFL5gypf8tLA</td>
      <td>I have walked by the Tokyo Hotel countless times. It reminds me of the type place that Peter Parker lives in in the Spiderman movies, Or the sort of place Marla Singer might live in, or maybe it is a carbon copy of the Hotel Zamenhof. Basically it is scuzzy and sleazy, through and through it is a fleabag, and I love it. Yesterday I finally walked in. I don't know why. Maybe because I just told my roommates and my landlord that I wouldn't be signing another lease, essentially leaving me "pre-homeless". Maybe because earlier that day I told boss that I'm quitting my job at the end of July, leaving me "pre-unemployed". Maybe because after all this time teetering between living on the edge of salvation and the brink of self destruction, I'm finally choosing my path, I'm hitting bottom. I'm giving up. Shit maybe I'm Nick Cage in Leaving Las Vegas, but where is my Elizabeth Shue? Who will look after me, who will walk me gently into that good night? If I choose to stay in this city, the city I love, then I may just stay in the "Hotel Tokyo". Because if I choose to stay, and that is a big if, then I'm going to need to stay somewhere cheap, with like-minded vagabonds, vagrants and naredowells. I'll need help, I'll need guidance.</td>
      <td>5</td>
      <td>12</td>
      <td>14</td>
      <td>23</td>
      <td>N</td>
      <td>tQfLGoolUMu2J0igcWcoZg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9/16/2010</td>
      <td>Ol</td>
      <td>nf3q2h-kSQoZK2jBY92FOg</td>
      <td>If you are considering staying here, watch this first: http://www.youtube.com/w…</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>6</td>
      <td>N</td>
      <td>tQfLGoolUMu2J0igcWcoZg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/5/2010</td>
      <td>i4HIAcNTjabdpG1K4F5Q2g</td>
      <td>Sb3DJGdZ4Rq__CqxPbae-g</td>
      <td>This place is disgusting, absolutely horrible, it is my second stay here, there is a strange stain on the other side of the bed I stay on, and my friends and I are rendered air conditionless, but it is 55 for downtown Chicago, what am I suppose to do?</td>
      <td>3</td>
      <td>11</td>
      <td>4</td>
      <td>9</td>
      <td>N</td>
      <td>tQfLGoolUMu2J0igcWcoZg</td>
    </tr>
  </tbody>
</table>
</div>



    Corpus contains 688329 documents (i.e. individual reviews)
    Corpus contains 10 columns
    




    Index(['date', 'reviewID', 'reviewerID', 'reviewContent', 'rating',
           'usefulCount', 'coolCount', 'funnyCount', 'flagged', 'hotelID'],
          dtype='object')



# Exploratory Data Analysis


![png](index_files/index_22_0.png)


The dataset is hugely imbalanced. This can be remedied by taking equal-sized samples for our sentiment analysis models.


![png](index_files/index_26_0.png)


# TextBlob (lexicon-based)


![png](index_files/index_36_0.png)



![png](index_files/index_37_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score_raw</th>
      <th>score_clean</th>
      <th>label_raw</th>
      <th>label_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.348153</td>
      <td>0.348153</td>
      <td>positive</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.120652</td>
      <td>0.113043</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.595</td>
      <td>0.595</td>
      <td>positive</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.160119</td>
      <td>0.152976</td>
      <td>negative</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.137459</td>
      <td>0.124528</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>0.1003</td>
      <td>0.130338</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>996</th>
      <td>-0.0625</td>
      <td>-0.05</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0.159694</td>
      <td>0.159694</td>
      <td>negative</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0.072379</td>
      <td>0.069451</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0.00928</td>
      <td>0.00928</td>
      <td>negative</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>



# Transformers (Transfer Learning)

    No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
    All model checkpoint layers were used when initializing TFDistilBertForSequenceClassification.
    
    All the layers of TFDistilBertForSequenceClassification were initialized from the model checkpoint at distilbert-base-uncased-finetuned-sst-2-english.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFDistilBertForSequenceClassification for predictions without further training.
    

The BERT implementation cannot handle text with more than 512 tokens. Thus, we set truncation to True and take only the first 512 tokens, cutting off extremely long reviews. This should not have a significant effect on our results.


![png](index_files/index_47_0.png)



![png](index_files/index_50_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tf_label_raw</th>
      <th>tf_conf_raw</th>
      <th>tf_label_clean</th>
      <th>tf_conf_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POSITIVE</td>
      <td>0.999790</td>
      <td>POSITIVE</td>
      <td>0.999740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POSITIVE</td>
      <td>0.999229</td>
      <td>POSITIVE</td>
      <td>0.998164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POSITIVE</td>
      <td>0.997314</td>
      <td>NEGATIVE</td>
      <td>0.954587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NEGATIVE</td>
      <td>0.954427</td>
      <td>NEGATIVE</td>
      <td>0.982705</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POSITIVE</td>
      <td>0.989587</td>
      <td>POSITIVE</td>
      <td>0.989647</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>NEGATIVE</td>
      <td>0.629143</td>
      <td>POSITIVE</td>
      <td>0.907640</td>
    </tr>
    <tr>
      <th>996</th>
      <td>NEGATIVE</td>
      <td>0.992209</td>
      <td>NEGATIVE</td>
      <td>0.992441</td>
    </tr>
    <tr>
      <th>997</th>
      <td>POSITIVE</td>
      <td>0.874674</td>
      <td>NEGATIVE</td>
      <td>0.856546</td>
    </tr>
    <tr>
      <th>998</th>
      <td>NEGATIVE</td>
      <td>0.987804</td>
      <td>NEGATIVE</td>
      <td>0.997394</td>
    </tr>
    <tr>
      <th>999</th>
      <td>NEGATIVE</td>
      <td>0.988836</td>
      <td>NEGATIVE</td>
      <td>0.994800</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>



# Model evaluation




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>True Negative</th>
      <th>False Negative</th>
      <th>False Positive</th>
      <th>True Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TextBlob on Raw Data</th>
      <td>358</td>
      <td>149</td>
      <td>142</td>
      <td>351</td>
    </tr>
    <tr>
      <th>TextBlob on Cleaned Data</th>
      <td>293</td>
      <td>88</td>
      <td>207</td>
      <td>412</td>
    </tr>
    <tr>
      <th>Transformer on Raw Data</th>
      <td>292</td>
      <td>54</td>
      <td>208</td>
      <td>446</td>
    </tr>
    <tr>
      <th>Transformer on Cleaned Data</th>
      <td>330</td>
      <td>95</td>
      <td>170</td>
      <td>405</td>
    </tr>
  </tbody>
</table>
</div>




![png](index_files/index_58_0.png)


We want to capture as much of the negative reviews as possible. TextBlob on Raw Data has the highest number of true negatives and the least false positives. However, this comes at the expense of more false negatives.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TextBlob on Raw Data</th>
      <th>TextBlob on Cleaned Data</th>
      <th>Transformer on Raw Data</th>
      <th>Transformer on Cleaned Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Accuracy</th>
      <td>0.71</td>
      <td>0.70</td>
      <td>0.74</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>Precision</th>
      <td>0.71</td>
      <td>0.72</td>
      <td>0.76</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.71</td>
      <td>0.70</td>
      <td>0.74</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.71</td>
      <td>0.70</td>
      <td>0.73</td>
      <td>0.73</td>
    </tr>
  </tbody>
</table>
</div>



We get evaluation metric scores higher than 70% across the board. Since we used unsupervised learning models, i.e.: did not train on the Yelp review data, these scores may indicate good performance.


![png](index_files/index_64_0.png)



![png](index_files/index_65_0.png)



![png](index_files/index_66_0.png)



![png](index_files/index_67_0.png)



![png](index_files/index_68_0.png)



![png](index_files/index_69_0.png)



![png](index_files/index_70_0.png)



![png](index_files/index_71_0.png)





