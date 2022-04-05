from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions

class ExtractiveQnA: 
    def __init__(self) -> None:
        self.model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name) 


    def get_answer (self, question: str, context: str) -> str:
        """
        Get answer from question and context. 
        """
        QA_input = {
            'question': question,
            'context': context
        }
        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        res = nlp(QA_input)
        return res.answer()
    
    def prepare_data_for_qna(self, df) -> str:
        text = df['text']
        # combine all comments into one string
        text =  ' '.join(text)
        return text
    