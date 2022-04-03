from zenml.pipelines import pipeline

@pipeline(enable_cache=False)
def data_pipeline(get_vid_links, raw_data_step, intermediate_data_step, final_data_step): 
    '''
    TODO 
    '''   
    link = get_vid_links() 
    raw_data = raw_data_step(link)
    intermediate_data = intermediate_data_step()
    final_data = final_data_step()
    return final_data  

