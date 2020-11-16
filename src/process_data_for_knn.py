
from utils import  is_array_col
import pandas as pd
from pathlib import Path
import numpy as np
from shape import Shape

from sklearn import preprocessing




def sample_normalizer(df,exclude,scaler,array_columns,array_lengths,divide_distributions):
    '''Normalize feature value ranges and divide histogram bins'''
    df= df.drop(exclude,axis=1)
    x = df.select_dtypes(include=np.number)
    if scaler == 'minmax':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(x)
    elif scaler == 'standardize':
        scaler = preprocessing.StandardScaler()
        scaler.fit(x)
    else:
        pass
        #print(f'Using provided scaler {scaler}')
    
    x_scaled=scaler.transform(x)
    
    df[x.columns] = x_scaled
    if divide_distributions:
        for length , array_name in zip(array_lengths,array_columns):
            for col in df.columns:
                if is_array_col(array_columns,col)==array_name:
                    df[col] = df[col]/length
    return df, scaler
    
    


def process_dataset_for_knn(data_path,divide_distributions,scaler = 'minmax'):
    '''Process data so that it is in form ready to be passed to knn interface'''
    data_path = Path(data_path)
    df = pd.read_csv(data_path,index_col=0)

   
    


    #Drop numerical features not relevant for similarity
    exclude = ['n_quads','n_vertices']
   

   
    



    array_columns = ["bounding_box",
                    "angle_three_vertices","barycenter_vertice", "two_vertices",
                    "square_area_triangle", "cube_volume_tetrahedron" ]


 
    
    


    #Get array lengths, replace...isdigit to make sure not to count things like bounding_box_volume as part of bounding_box array.
    array_lengths = [len([x for x in df.columns if is_array_col(array_columns,x)==y]) for y in array_columns]

    non_numeric_columns = ["file_name","classification"]


    single_numeric_columns = [ x for x in set(df.columns)-set(non_numeric_columns) if not is_array_col(array_columns,x)]


    assert len(single_numeric_columns) +len(non_numeric_columns) + sum(array_lengths) == df.shape[1], "Column counts may be incorrect!"

    for col in single_numeric_columns:
        df[col].fillna(df[col].median(),inplace=True)


    #remove extreme outliers
    df = df[(df[single_numeric_columns]<=df[single_numeric_columns].quantile(0.999)).all(axis=1)]
    df = df[(df[single_numeric_columns]>=df[single_numeric_columns].quantile(0.001)).all(axis=1)]
    df.reset_index(inplace=True,drop=True) #Reset index after removing outliers


    #Pass through normalizer, so scale and divide arrays as needed for knn. Use same for query
    df , scaler = sample_normalizer(df,exclude,scaler,array_columns,array_lengths,divide_distributions=divide_distributions)

    
    return df, exclude, scaler, array_columns, array_lengths


if __name__ == '__main__':
    data_path = Path("processed_data/data_coarse1_processed_10000_10000000.0.csv")
    a = process_dataset_for_knn(data_path,divide_distributions=True)
   





