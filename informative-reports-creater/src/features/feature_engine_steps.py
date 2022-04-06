import pandas as pd

from zenml.steps import step, Output
from .build_features import FeatureEngineering 

@step 
def basic_feature_processing(data: pd.DataFrame) -> Output(
       ouput_data = pd.DataFrame
):
       feature_engine = FeatureEngineering(data)
       data = feature_engine.remove_features()
       return data 


@step
def vectorization(data: pd.DataFrame) -> Output(
       ouput_data = pd.DataFrame
): 
    column = 'text'
    data_copy = data.copy() 
    feature_engine = FeatureEngineering(data_copy)
    data_copy = feature_engine.tf_idf_vectorizer()
    return data_copy


