from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import prophet as Prophet

from utils import prophet_trend_analysis

class ITrendMethod(ABC):
    """_summary_

    Args:
        meta (_type_, optional) : Defaults to ABCMeta.
    """
    def __init__(self, successor = None) -> None:
        self.successor = successor

    @abstractmethod
    def handle_request(self):
        pass

    @abstractmethod
    def transform():
        """Transform the data into suitable format for analysis
        """
        pass

    @abstractmethod
    def analysis():
        """Implementation of the Model
        """
        pass

    @abstractmethod
    def infer():
        """_summary_
        """
        pass

    @abstractmethod
    def sentence_maker():
        pass
    


class LinearMethod(ITrendMethod):
    """ _summary_

    Args:
        TrendMethod (_type_): _description_
    """
    def __init__(self):
        super().__init__()

    def handle_request(self, df, date_column, target_column, kpi_name):
        return self.sentence_maker(df, date_column, target_column ,kpi_name)
        
    def transform(self, df: pd.DataFrame, date_column: str, target_column: str):
        training_df = df[[date_column, target_column]]
        x_var = np.arange(len(df)).reshape(-1)
        training_df.loc[:, 'x'] = list(x_var)
        y_var = training_df[target_column].values.reshape(-1)

        return training_df, x_var.reshape(-1, 1), y_var

    def analysis(self, df: pd.DataFrame, date_column: str, target_columnn: str):
        lr_df, X, y = self.transform(df, date_column, target_columnn)
        lin_reg_model = LinearRegression()
        lr_result = lin_reg_model.fit(X=X.reshape(-1, 1), y= y)

        return X, y, lr_result, lr_df

    def infer(self, df: pd.DataFrame, date_column: str, target_columnn: str):

        X, y, lr_model, lr_df = self.analysis(df, date_column, target_columnn)
        slope = lr_model.coef_[0]
        first_point = y[0]
        last_point = y[-1]
        first_lr_point = lr_model.predict(X[0].reshape(-1, 1))[0]
        last_lr_point = lr_model.predict(X[-1].reshape(-1, 1))[0]
        result_dict = {'lr_model': lr_model, 'slope':slope, 'first_point':first_point, 'last_point':last_point,
                       'first_lr_point':first_lr_point, 'last_lr_point':last_lr_point}
        result_dict['percentage_change'] = (result_dict['last_point'] - result_dict['first_point'])*100/result_dict['last_point']
        result_dict['max'] = y.max()
        result_dict['min'] = y.min()
        return result_dict
    
    def sentence_maker(self, df:pd.DataFrame, date_column: str, target_column:str, kpi_name:str):
        result_dict = self.infer(df, date_column, target_column)
        if result_dict['slope'] > 0 and result_dict['percentage_change'] > 0:
            return f"The current kpi {kpi_name} is in increasing trend from {result_dict['first_lr_point']} with {round(result_dict['percentage_change'], 1)}% to {result_dict['last_lr_point']}."
        elif result_dict['slope'] < 0 and result_dict['percentage_change'] < 0:
            return f"The current kpi {kpi_name} is in decreasing trend from {result_dict['first_lr_point']} with {round(result_dict['percentage_change'], 1)}% to {result_dict['last_lr_point']}."
        else:
            return f"The current kpi {kpi_name} is consolidating between {result_dict['max']} and {result_dict['min']}"

    

class ProphetMethod(ITrendMethod):
    def __init__(self, successor):
        super().__init__(successor=successor)

    def handle_request(self, df:pd.DataFrame, date_column: str, target_column:str, kpi_name ):
        x = self.infer(df, date_column, target_column)
        if isinstance(x,str):
            return self.successor.handle_request(df, date_column, target_column, kpi_name)
        else:
            return self.sentence_maker(df, date_column, target_column, kpi_name)
        
    def transform(self, df:pd.DataFrame, date_column: str, target_column:str):
        training_df = df[[date_column, target_column]]
        training_df.columns = ['ds', 'y']
        training_df['ds'] = pd.to_datetime(training_df['ds'])
        return training_df

    def analysis(self, df:pd.DataFrame, date_column: str, target_column:str):
        df_train = self.transform(df, date_column, target_column)
        cpd_list, lr_models = prophet_trend_analysis(df_train)
        return df_train, cpd_list, lr_models

    def infer(self, df:pd.DataFrame, date_column: str, target_column:str):
        trained_df, change_points, lin_reg_models = self.analysis(df, date_column, target_column)
        if len(change_points) > 2:
            result_dict = {
                            'last_section_lr' : lin_reg_models[f'section{len(change_points)-2}'],
                            'last_change_point_index' : change_points[-2],
                            'last_change_point_value' : trained_df.loc[change_points[-2], 'y'],
                            'last_change_point_date' : trained_df.loc[change_points[-2], 'ds'],
                            'last_point_index' : change_points[-1]-1,
                            'last_point_value' : trained_df.loc[change_points[-1]-1, 'y'],
                            'last_change_date' : trained_df.loc[change_points[-1]-1, 'ds']
                            }
            result_dict['percentage_change'] = (result_dict['last_point_value'] - result_dict['last_change_point_value'])*100/result_dict['last_point_value']
            result_dict['last_section_max'] = trained_df.loc[result_dict['last_change_point_index']:result_dict['last_point_index'], 'y'].max()
            result_dict['last_section_min'] = trained_df.loc[result_dict['last_change_point_index']:result_dict['last_point_index'], 'y'].min()
            return result_dict
        else:
            return 'No Change Points Detected'
        
    def sentence_maker(self, df:pd.DataFrame, date_column: str, target_column:str, kpi_name:str):
        result_dict = self.infer(df, date_column, target_column)
        if result_dict['last_section_lr'].coef_[0] > 0 and result_dict['percentage_change'] > 0:
            return f"The current kpi {kpi_name} has increased {round(result_dict['percentage_change'], 1)}% since {result_dict['last_change_point_date'].strftime('%d-%m-%Y')} from {result_dict['last_change_point_value']} to {result_dict['last_point_value']}."
        elif result_dict['last_section_lr'].coef_[0] < 0 and result_dict['percentage_change'] < 0:
            return f"The current kpi {kpi_name} has decreased {round(result_dict['percentage_change'], 1)}% since {result_dict['last_change_point_date'].strftime('%d-%m-%Y')} from {result_dict['last_change_point_value']} to {result_dict['last_point_value']}."
        else:
            return f"The current kpi {kpi_name} is consolidating between {result_dict['last_section_max']} and {result_dict['last_section_min']}"


class trendContext():
    def get_model(self):
        return ProphetMethod(LinearMethod())
    
if __name__ == '__main__':
    model = trendContext().get_model()
    # x = model.handle_request()
    print(type(model))
