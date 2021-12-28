from sklearn.utils import resample

import pandas as pd

def hello_func(char):
    print('hello')
    return('hello')

def rebalance_classes(input_df):
    class_1 = input_df.loc[input_df.fetal_health == '1']
    class_2 = input_df.loc[input_df.fetal_health == '2']
    class_3 = input_df.loc[input_df.fetal_health == '3']
    
    class_2 = resample(class_2, n_samples = class_1.shape[0])
    class_3 = resample(class_3, n_samples = class_1.shape[0])
    
    output_df = pd.concat([class_1, class_2, class_3], ignore_index = True)
    return(output_df)

def accuracy_by_class(predicted, actual):
    out_vector = []
    out_vector.append(np.mean(predicted == actual))
    for i in range(1, 4):
        out_vector.append(np.mean(predicted[actual == str(i)] == str(i)))
    out_vector.append(np.mean(out_vector[1:4]))
    out_vector.append(f1_score(y_true = actual, y_pred = predicted, average = "macro"))
    return(out_vector)
