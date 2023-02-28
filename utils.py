import pandas as pd

def get_metrics(metrics_dict, filt_df_dict):
    result_dict = dict()
    result_dict['Filter'] = []
    for metric_name in metrics_dict:
        result_dict[metric_name] = []
    
    for filt_df_name in filt_df_dict:
        result_dict['Filter'].append(filt_df_name)

        filt_df = filt_df_dict[filt_df_name]
        y_true = filt_df['True']
        y_pred = filt_df.drop('True', axis=1).iloc[:, 0]        
        for metric_name in metrics_dict:
            metric_func = metrics_dict[metric_name]
            result_dict[metric_name].append(
                metric_func(y_true, y_pred)
            )
    
    return pd.DataFrame(result_dict)