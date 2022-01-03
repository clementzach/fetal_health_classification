def import_data(csv_path):
    import pandas as pd

    fetal_health = pd.read_csv(csv_path)

    fetal_health['histogram_tendency'] = fetal_health['histogram_tendency'].astype(str)
    fetal_health = pd.get_dummies(fetal_health)
    fetal_health.fetal_health = fetal_health.fetal_health.astype(int).astype(str) #make outcome categorical
    return(fetal_health)

def rebalance_classes(input_df):

    from sklearn.utils import resample

    import pandas as pd 
    class_1 = input_df.loc[input_df.fetal_health == '1']
    class_2 = input_df.loc[input_df.fetal_health == '2']
    class_3 = input_df.loc[input_df.fetal_health == '3']
    
    class_2 = resample(class_2, n_samples = class_1.shape[0])
    class_3 = resample(class_3, n_samples = class_1.shape[0])
    
    output_df = pd.concat([class_1, class_2, class_3], ignore_index = True)
    return(output_df)

def accuracy_by_class(predicted, actual):
    from sklearn.metrics import f1_score

    import numpy as np
    import pandas as pd
    out_vector = []
    out_vector.append(np.mean(predicted == actual))
    for i in range(1, 4):
        out_vector.append(np.mean(predicted[actual == str(i)] == str(i)))
    out_vector.append(np.mean(out_vector[1:4]))
    out_vector.append(f1_score(y_true = actual, y_pred = predicted, average = "macro"))
    return(out_vector)


def print_metrics(final_accuracy):
    from os import linesep
    print_str = "Overall accuracy: {}".format(final_accuracy[0]) + linesep

    for i in range(1,4):
        print_str = print_str + "Accuracy for class {}: {}".format(i, final_accuracy[i]) + linesep
    
    print_str = print_str + "Weighted Accuracy: {}".format(final_accuracy[4]) + linesep

    print_str = print_str + "Macro F1 score: {}".format(final_accuracy[5]) + linesep
    return(print_str)
