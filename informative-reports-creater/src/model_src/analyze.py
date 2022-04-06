import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS 
from transformers import pipeline

class AnalyzeComments: 
    def __init__(self, data) -> None:
        self.data = data 

    def word_cloud_of_comments(self): 
        """
        Generates a word cloud of the comments. 
        """
        # Create a word cloud of the comments. 
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              min_font_size=10).generate(str(self.data['text']))

        # Display the generated image:
        plt.figure(figsize=(10, 8), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()  
    
    
    def top_n_words_in_comments(self, top_n: int = 10):
        """
        Generates a frequency plot of the top n words in the comments. 
        """
        # Create a frequency plot of the top n words in the comments.  
        # remove stop words
        stop_words = set(STOPWORDS)
        self.data['text'] = self.data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))   
        top_words = self.data['text'].str.split().apply(pd.Series, 1).stack().value_counts()[:top_n]
        top_words.plot(kind='barh', figsize=(10, 8), color='#008080')
        plt.title('Top {} words in comments'.format(top_n))
        plt.xlabel('Frequency')
        plt.ylabel('Word')
        plt.show()

    def overview_of_video_good_or_not(self):
        """
        Generates a frequency plot of the top n words in the comments. 
        """
        # Create a frequency plot of the top n words in the comments. 
        sentiment_pipeline = pipeline("sentiment-analysis")
        self.data['sentiment'] = self.data['text'].apply(sentiment_pipeline) 
        print(self.data)
    def top_n_comments_by_sentiment(self, top_n: int = 10):
        """
        Generates a frequency plot of the top n words in the comments. 
        """
        # Create a frequency plot of the top n words in the comments. 
        self.data['sentiment'].value_counts()[:top_n].plot(kind='barh', figsize=(10, 8), color='#008080')
        plt.title('Top {} comments by sentiment'.format(top_n))
        plt.xlabel('Frequency')
        plt.ylabel('Sentiment')
        plt.show()
    
    def top_n_comments_by_vote(self, top_n: int = 10):
        """
        Generates a frequency plot of the top n words in the comments. 
        """
        # Create a frequency plot of the top n words in the comments. 
        self.data['votes'].value_counts()[:top_n].plot(kind='barh', figsize=(10, 8), color='#008080')
        plt.title('Top {} comments by vote'.format(top_n))
        plt.xlabel('Frequency')
        plt.ylabel('Vote')
        plt.show()
    
    def get_top_n_important_feedback(self, top_n: int = 10): 
        """ 
        TODO : I need to do data collection and do labelling and then make use of Semi Supervised learning to train a model which gives important feedbacks and etc.
        """
        pass 
