import pandas as pd
import numpy as np
import csv
import os.path

def readData():
    # Reads the Iris Dataset from the csv file in utils
    # Last column (4) contains the label information
    # 1 = Iris-setosa
    # 2 = Iris-versicolor
    # 3 = Iris-virginica

    data = np.empty([150, 5])
    i = 0
    with open('utils/iris_data', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        for row in filereader:
            r = np.array(row)
            # Convert the rows from strings to floats
            r_float = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]

            data[i] = np.append(r_float, int(i / 50 + 1))

            i += 1
    return data

def readCSV(src_filepath: str) -> pd.DataFrame:
    """
    Summary:
        Reads in a .csv file.

    Parameters:
        src_filepath (str): A filepath string to the dataset to split

    Returns:
        df (pd.DataFrame): A dataframe object.
    """
    
     # Validate input path
    if not os.path.isfile(src_filepath):
        raise Exception("src_filepath is not a file.")

    # Read input .csv
    data = pd.read_csv(src_filepath, header=1, index_col=0)
    df = pd.DataFrame(data)
    return df

def splitAndEnumData(dataFrame: pd.DataFrame, labelMap: dict) -> tuple[np.array, np.array]:
    """
    Summary:
        Splits a dataset into x and y. Enumerates the classes based on the mapping provided.

        1 = "Iris-setosa"
        2 = "Iris-versicolor"
        3 = "Iris-virginica"

    Parameters:
        dataFrame (pd.DataFrame): The dataset to split
        labelMap (dict): The mapping from label to desired integer (i.e., {<label>:<int>}). If a label is not given an integer,
                         it, along with the corresponding sample is remove entirely.

    Returns:
        x (np.array): Samples.
        y (np.array): Enumerated labels.
    """
    # Remove rows that do not have a label map or remap the label
    labelsToMap = labelMap.keys()
    oldData = dataFrame.to_numpy()
    newData = np.empty((0,oldData.shape[1]),int)

    # Iterate through all rows and check last column
    for row_idx in range(len(oldData)):
        current_label = oldData[row_idx,-1]
        if current_label in labelsToMap:
            temp = oldData[row_idx]
            temp[-1] = labelMap[current_label]
            newData = np.append(newData,[temp],axis=0)
    return newData[:,:-1], newData[:,-1].astype(np.int_)

def normalizeData(x) -> np.array:
    # Normalize each feature (column)
    for i in range(len(x[0,:])):
        x_col = x[:,i]
        x[:,i] = (x_col-x_col.min()) / (x_col.max()-x_col.min())
    return x


def splitDataToCSV(src_filepath: str, training_size=0.50, validation_size=0.00) -> None:
    """
    Summary:
        Splits a dataset and places the training, validation, and testing sets into separate .csv files according
        to the sizes specified by the input params. The size of the testing set is determined by the remaining 
        datapoints not assigned to either the training nor validation set. The sizes of the training and validation sets
        are defined as ratios (0.00, 1.00)

    Parameters:
        src_filepath (str): A filepath string to the dataset to split
        training_size (float): A ratio which determines the training set size (default is 0.50)
        validation_size (float): A ratio which determines the validation set size (default is (0.00)

    Returns:
        None
    """

    # Validate input path
    if not os.path.isfile(src_filepath):
        raise Exception("src_filepath is not a file.")
    
    # Reject whole number sizes
    if (training_size >= 1.0) or (validation_size >= 1.0):
        raise Exception("training_size and validation_size must be ratios.")

    # Read input .csv
    data = pd.read_csv(src_filepath, header=0) # include header row
    df = pd.DataFrame(data)
    num_samples = len(df.index)

    # Get test_size and validate
    test_size = num_samples - round(num_samples*training_size) - round(num_samples*validation_size)
    if test_size <= 0:
        raise Exception("test_size is invalid! Please correct training_size and validation_size.")
    
    # Get unique class labels
    classes = df[df.columns[-1]].unique()

    # Group rows by class label then split each class by the ratios specified
    training_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    testing_data = pd.DataFrame()
    for idx, c in enumerate(classes):
        # Get first dataframe grouped by class
        temp_df = df[df[df.columns[-1]] == c]

        # Randomly sample dataset and place into training, validation, and testing files
        temp_training_data = temp_df.sample(frac=training_size)
        temp_df = temp_df.drop(temp_training_data.index)
        temp_validation_data = temp_df.sample(frac=validation_size)
        temp_testing_data = temp_df.drop(temp_validation_data.index)

        # See if we need initialize dataframe or append to create dataframe to be saved
        if(idx == 0):
            training_data = temp_training_data
            validation_data = temp_validation_data
            testing_data = temp_testing_data
        # Append to existing set
        else:
            training_data = pd.concat([training_data, temp_training_data])
            validation_data = pd.concat([validation_data, temp_validation_data])
            testing_data = pd.concat([testing_data, temp_testing_data])

    # Write data to their respective .csv files
    directory, filename = os.path.split(src_filepath)
    filename, file_extension = os.path.splitext(filename)
    training_data.to_csv(directory + "/" + filename + "_training" + file_extension, index=False)
    validation_data.to_csv(directory + "/" + filename + "_validation" + file_extension, index=False)
    testing_data.to_csv(directory + "/" + filename + "_testing" + file_extension, index=False)

    return