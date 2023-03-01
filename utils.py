import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from metrics import snr, psnr

def get_metrics(df_dict, index_column=None, metrics_dict=None):
    if metrics_dict is None:
        metrics_dict = metrics_dict = {
            'mean_squared_error': mean_squared_error,
            'R-squared': r2_score,
            'snr': snr,
            'psnr': psnr
        }

    result_dict = dict()
    if index_column is None:
        result_dict['Filter'] = []
    else:
        result_dict[index_column] = []
    
    for metric_name in metrics_dict:
        result_dict[metric_name] = []
    
    for filt_df_name in df_dict:
        result_dict[index_column].append(filt_df_name)

        filt_df = df_dict[filt_df_name]
        y_true = filt_df['True']
        y_pred = filt_df.drop('True', axis=1).iloc[:, 0]        
        for metric_name in metrics_dict:
            metric_func = metrics_dict[metric_name]
            result_dict[metric_name].append(
                metric_func(y_true, y_pred)
            )
    
    return pd.DataFrame(result_dict)