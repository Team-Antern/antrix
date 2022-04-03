import pandas as pd 
# from .data.raw.raw_data_ingest import main
# from .data.interim.interm_data_process import IntermediateDataProcess  
# from .data.processed.final_process import FinalProcessing 

from .src.data.make_dataset_pipeline import data_pipeline 
from .src.data.data_steps import raw_data_step, intermediate_data_step, final_data_step, get_vid_links, PreTrainingConfigs

def run_training(): 
    training = data_pipeline( 
        get_vid_links(),
        raw_data_step(),
        intermediate_data_step(),
        final_data_step(),
    )

    training.run()

if __name__ == "__main__":
    run_training()
