import pandas as pd

from zenml.steps import step, Output
from zenml.steps import BaseStepConfig

from ...data.raw.raw_data_ingest import main
from ...data.interim.interm_data_process import IntermediateDataProcess 
from ...data.processed.final_process import FinalProcessing

class PreTrainingConfigs(BaseStepConfig):
      raw_data_path: str = "informative-reports-creater/data/raw/yt_comments.csv" 
      intermediate_data_path: str = "informative-reports-creater/data/interim/interm_data.csv" 
      final_data_path: str = "informative-reports-creater/data/processed/final_data.csv"
      

@step 
def get_vid_links() -> Output( 
      link = str 
): 
      link = "https://www.youtube.com/watch?v=NWONeJKn6kc" 
      return link 
@step 
def raw_data_step(link: str = "https://www.youtube.com/watch?v=NWONeJKn6kc") -> Output( 
      raw_data = pd.DataFrame
):    
      raw_data = main(link)
      return raw_data

@step 
def intermediate_data_step() -> Output( 
      intermediate_data = pd.DataFrame
):  
      raw_data_path: str= "informative-reports-creater/data/raw/yt_comments.csv"
      intermediate_data = IntermediateDataProcess(raw_data_path)
      interim_data = intermediate_data.main()  
      return interim_data

@step
def final_data_step() -> Output(
      final_data = pd.DataFrame
):
      intermediate_data_path: str = "informative-reports-creater/data/interim/interm_data_process.csv"
      final_data = FinalProcessing(intermediate_data_path)
      final_data = final_data.main()
      return final_data 