from transformers import AutoModelForQuestionAnswering, pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

# model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
# a) Get predictions

class ExtractiveQnA: 
    def __init__(self) -> None:
        self.nlp = pipeline('question-answering') 


    def get_answer (self, question: str, context: str) -> str:
        """
        Get answer from question and context. 
        """
        QA_input = {
            'question': question,
            'context': context
        }
        res = self.nlp(QA_input)
        return res
    
    def prepare_data_for_qna(self, df) -> str:
        text = df['text']
        # combine all comments into one string 
        #$ convert every row to str and remove any floats and put full stop every row  
        # add a full stop in every text 
        text = ' '.join(str(x) for x in text)
        return text

