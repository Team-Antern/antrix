from zenml.pipelines import pipeline

@pipeline(enable_cache=False)
def feature_engineering_pipeline(read_data,basic_feature_processing, add_features, vectorization): 
    '''
    TODO 
    '''  
    data = read_data()
    data = basic_feature_processing(data)
    data = add_features(data)
    data = vectorization(data, 'text')
    return data

