import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

def dbscan_1d(points):
    """_summary_

    Args:
        points (_type_): _description_

    Returns:
        _type_: _description_
    """
    dbscan = DBSCAN( eps = 10, min_samples = 2)
    labels = dbscan.fit_predict(points)

    # Create lists for each cluster
    unique_labels = np.unique(labels)
    clusters = {label: points[labels == label].flatten() for label in unique_labels if label != -1}

    for label, cluster_points in clusters.items():
        clusters[label] =  int(cluster_points.mean())
    return list(clusters.values())

def piece_wise_linear_reg(X,y, result):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        result (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(y)-1 not in result:
        np.append(result, len(y)-1)
    if 0 not in result:
        result = [0] + result

    # Add colored backgrounds for each segment
    color_list = []
    lr_models = {}
    for i in range(len(result) - 1):
        segment_X = X[result[i]:result[i + 1]]
        segment_y = y[result[i]:result[i + 1]]

        # Fit linear regression model
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(segment_X, segment_y)
        lr_models[f'section{i}'] = lin_reg_model
        # Get slope to determine the trend
        slope = lin_reg_model.coef_

        # Set background color based on the trend
        color = "blue" if -0.1 < slope < 0.1 else "green" if slope > 0 else "red" if slope < 0 else None
        color_list.append(color)

    return color_list, result, lr_models

def prophet_trend_analysis(data_with_time_df,  interval_width = 0.99, changepoint_range = 1):
    """_summary_

    Args:
        data_with_time_df (_type_): _description_
        interval_width (float, optional): _description_. Defaults to 0.99.
        changepoint_range (int, optional): _description_. Defaults to 1.
    """
    prophet_model = Prophet(changepoint_range = changepoint_range, interval_width = interval_width)
    prophet_model = prophet_model.fit(data_with_time_df)
    #dates of the change points
    signif_changepoints = prophet_model.changepoints[
                            np.abs(np.nanmean(prophet_model.params['delta'], axis=0)) >= 0.01
                                ] if len(prophet_model.changepoints) > 0 else []
    points = signif_changepoints.index.values.reshape(-1, 1)
    
    if len(points) > 3:
        change_points_index = dbscan_1d(points)
    else:
        change_points_index = list(points.reshape(-1))
  
    # print(signif_changepoints)
    # print(change_points_index)
    data_y = data_with_time_df['y'].values
    data_X = np.arange(len(data_y)).reshape(-1, 1)
    change_points_index.append(len(data_y))
    color_list, result_cpd, _ = piece_wise_linear_reg(data_X, data_y, change_points_index)
    # for _ in range(3):
    #     result_cpd, color_list = merge_consecutive_trends(indices= result_cpd, trends= color_list)
    #     color_list, result_cpd, _ = piece_wise_linear_reg(data_X, data_y, result_cpd)
    _, result_cpd, lr_models = piece_wise_linear_reg(data_X, data_y, result_cpd)

    return result_cpd, lr_models