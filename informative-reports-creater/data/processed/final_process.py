import numpy as np 
import pandas as pd 
import demoji
import logging 

class FinalProcessing: 
    def __init__(self, df): 
        self.df = df

    def main(self): 
        logging.info("Applying main method on  comment")
        try:
            self.df['text'] = self.df["text"].apply( 
                lambda x: self.remove_punctuation(x) 
            ) 
            self.df.to_csv(
                "informative-reports-creater/data/processed/processed_data.csv",
                index=False
            )  
            return self.df 
        except Exception as e:
            logging.error(
                "Error in applying main method on  comment"
            )
            logging.error(e) 

    def emoji_to_text(self, text): 
        try: 
            logging.info("Applying emoji_to_text on the data started")
            demoji.download_codes()  
            emojival = demoji.findall(text) 
            # check if the dict is empty
            if len(text.keys()) > 0: 
                # replace emoji with value if key
                # get the value of the key 
                emojival = [text[key] if key in text.keys() else key for key in text] 
                # join the list of values
                # join comment text and emoji value
                text =  emojival + [text] 
            print(text)
            return text
            print(text)
        except Exception as e: 
            logging.error(
                "Error in applying emoji_to_text methods on  data"
            )
            logging.error(e)
            return None

    
    def remove_punctuation(self, comment):
        logging.info("Applying remove_punctuation methods on  comment")
        try:
            comment = comment.split()
            comment = [word for word in comment if word.isalpha()]
            comment = " ".join(comment)
            return comment
        except Exception as e:
            logging.error(
                "Error in applying remove_punctuation method on comment"
            )
            logging.error(e) 



