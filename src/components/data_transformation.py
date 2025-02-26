import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns=['reading score','writing score']
            categorical_columns=['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('std_scaler',StandardScaler())
                ]
            )

            logging.info("Numerical pipeline created")

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore')),
                    ('std_scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def init_data_tranformation(self,train_data,test_data):
        try:
            train_data=pd.read_csv(train_data)
            test_data=pd.read_csv(test_data)

            logging.info("train and test data read successfully")

            preprocessing_obj=self.get_data_transformer_obj()
            logging.info("Preprocessing object created")

            target_column='math score'

            input_feature_train_df=train_data.drop(target_column,axis=1)
            target_feature_train_df=train_data[target_column]

            input_feature_test_df=test_data.drop(target_column,axis=1)
            target_feature_test_df=test_data[target_column]
            logging.info("Features and target split successfully")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("Data transformation completed successfully")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Train and test arrays created successfully")

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)