import pandas as pd 
# from .data.raw.raw_data_ingest import main
# from .data.interim.interm_data_process import IntermediateDataProcess  
# from .data.processed.final_process import FinalProcessing 

from .src.data.make_dataset_pipeline import data_pipeline  
from .src.features.make_feature_pipeline import feature_engineering_pipeline

from .src.data.data_steps import raw_data_step, intermediate_data_step, final_data_step, get_vid_links, PreTrainingConfigs, read_data
from .src.features.feature_engine_steps import basic_feature_processing, vectorization

from .src.model_src.qna_system_steps import get_answer, get_summary

from .src.model_src.analyze import AnalyzeComments
def run_training(): 
    data_pipelines = data_pipeline( 
        get_vid_links(),
        raw_data_step(),
        intermediate_data_step(),
        final_data_step()
    )

    data_pipelines.run()

    feature_enginering = feature_engineering_pipeline( 
        read_data(), 
        basic_feature_processing(),
        vectorization(),
    ) 

    feature_enginering.run() 

def analysis():  
    df = pd.read_csv(r"informative-reports-creater/data/processed/processed_data.csv") 
    df["text"].fillna('No word', inplace=True)
    analyze = AnalyzeComments(df) 
    analyze.word_cloud_of_comments() 
    analyze.top_n_words_in_comments(top_n=10)
    analyze.overview_of_video_good_or_not()
    analyze.top_n_comments_by_sentiment(top_n=10)
    analyze.top_n_comments_by_vote(top_n=10)
    
if __name__ == "__main__":
    run_training()
    analysis() 