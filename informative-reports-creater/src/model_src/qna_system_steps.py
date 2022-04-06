import json
from multiprocessing.connection import answer_challenge
import pandas as pd 
import json 
from .qna_system import ExtractiveQnA 
import transformers
from transformers import pipeline

def get_answer(question: str, data: pd.DataFrame) -> json: 

    qna = ExtractiveQnA()
    data_copy = data.copy()
    context = qna.prepare_data_for_qna(data_copy)  
    print(context)
    answer = qna.get_answer(question, context)  
    print(answer)
    return answer


def get_summary(data: pd.DataFrame) -> json: 
    qna = ExtractiveQnA()
    data_copy = data.copy()
    context = qna.prepare_data_for_qna(data_copy)  
    print(context)
    summarizer = pipeline("summarization")
    summarized = summarizer(context, min_length=75, max_length=300)
    return summarized