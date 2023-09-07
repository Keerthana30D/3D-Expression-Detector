import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn import preprocessing



def dataLoad(data_path):
    #To store the data, creating empty lists.
    input_data = []
    #To store the labels, creating empty lists.
    input_labels = []
    #To iterate the list of directories in the provided data path.
    for data_dir in os.listdir(data_path):
        #a path to the current subject directory is being created.
        input_path = os.path.join(data_path, data_dir)
        #To iterate the list of directories in the current subject directory.
        for file_dir in os.listdir(input_path):
            #a path to the current expression directory is being created.
            input_file_path = os.path.join(input_path, file_dir)
            #To iterate through the files in the current expression directory's list.
            for file_name in os.listdir(input_file_path):
                #To create a path to the current file
                file_path = os.path.join(input_file_path, file_name)
                #To open the file for reading
                with open(file_path, "r") as f:
                    #To store landmarks, creating an empty list 
                    in_landmarks = []
                    #To print the expression path and file name
                    print(input_file_path, file_name)
                    #executing the try on following lines of code
                    try:
                        #To iterate over the lines in the file
                        for line in f:
                            #To remove the line's first element after scanning the landmarks
                            in_landmark = list(map(float, line.strip().split()))[1:]
                            #To add the landmark to the list of landmarks
                            in_landmarks.append(in_landmark)
                        #To add the list of landmarks to the list of data
                        input_data.append(in_landmarks)
                        #To include the name of the expression directory in the list of labels
                        input_labels.append(file_dir)
                    #If an exception arises, handle it and move on to the loop's subsequent iteration.
                    except Exception as e:
                        continue
    #To convert the lists to Numpy arrays, and then return  
    return np.array(input_data), np.array(input_labels)

def cross_data_validation(X, y, data_clf, n_splits=10):
    #Initialize empty lists to serve as the expected labels and test indices.
    data_pred = []
    data_test_indices = []
    
    #To create k-folds for splitting the data
    kf = KFold(n_splits=n_splits)
    #To train the classifier on the training set
    for train_index, test_index in kf.split(X):
        data_clf.fit(X[train_index], y[train_index])
        #To predict the labels of the test set
        data_pred.append(data_clf.predict(X[test_index]))
        #To Save the test indices for later use
        data_test_indices.append(test_index)
    
    #To concatenate the predicted labels and test indices
    data_pred = np.concatenate(data_pred)
    test_indices = np.concatenate(data_test_indices)

    #To calculate the confusion matrix and evaluation metrics
    conf_mat_list = confusion_matrix(y[test_indices], data_pred)
    accuracy = accuracy_score(y[test_indices], data_pred)
    precision = precision_score(y[test_indices], data_pred, average='weighted', zero_division=0)
    recall = recall_score(y[test_indices], data_pred, average='weighted', zero_division=0)
    
    #To return the confusion matrix and evaluation metrics
    return conf_mat_list, accuracy, precision, recall

def get_data_classifier(classifier_name):
    #Returns an instance of the RandomForestClassifier if the name is "RF".
    if classifier_name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    #Returns an instance of the LinearSVC if the name is "SVM"
    elif classifier_name == "SVM":
        return LinearSVC(random_state=42)
    #Returns an instance of the DecisionTreeClassifier if the name is "TREE". 
    elif classifier_name == "TREE":
        return DecisionTreeClassifier(random_state=42)
    #Returns a invalid message if the name is not valid. 
    else:
        raise ValueError("Use valid classifier name")


def translate_data_landmarks(in_landmarks):
#To calculate the average landmark point across all samples
    avgData = np.mean(in_landmarks, axis=0)
    #To translate them, take the average and subtract it from each landmark and retun 
    return in_landmarks - avgData


def rotate_data_180_degrees(in_landmarks, data_axis):

"""
    Rotates the landmarks 180 degrees around the chosen axis.

    parameters: axis (str): rotation axis, must be 'x', 'y', or 'z'; landmarks (array): 3D array containing x, y, and z coordinates of landmarks

    Returns: an array of landmarks that have been 180 degrees rotated around the supplied axis, rotated_landmarks.

"""
    #To calculate the value of pi
    pi = round(2 * np.arccos(0.0), 3)
    #To check which axis to rotate around and calculate the rotation matrix
    if data_axis == 'x':
        data_rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(pi), np.sin(pi)],
                                    [0, -np.sin(pi), np.cos(pi)]])
    elif data_axis == 'y':
        data_rotation_matrix = np.array([[np.cos(pi), 0, -np.sin(pi)],
                                    [0, 1, 0],
                                    [np.sin(pi), 0, np.cos(pi)]])
    elif data_axis == 'z':
        data_rotation_matrix = np.array([[np.cos(pi), np.sin(pi), 0],
                                    [-np.sin(pi), np.cos(pi), 0],
                                    [0, 0, 1]])
    else:
        #To raise an error if the specified axis is invalid
        raise ValueError(f"Invalid axis '{data_axis}'. Must be 'x', 'y', or 'z'.")
    #To rotate the landmarks using the rotation matrix and to return the rotated landmarks  
    return np.dot(in_landmarks, data_rotation_matrix)



def run_prediction(classifier_name, data_type, data_path):
    #To load the data from the specified path
    input_data, input_labels = dataLoad(data_path)
    
    #Depending on the data type, it perform any appropriate transformations
    if data_type == "Original":
        pass
    elif data_type == "Translated":
        input_data = translate_data_landmarks(input_data)
    elif data_type in ("RotatedX", "RotatedY", "RotatedZ"):
        data_axis = data_type[7:].lower()
        input_data = rotate_data_180_degrees(input_data, data_axis)
    else:
        raise ValueError("Invalid data type")
 
    #To get the classifier object based on the specified name 
    data_classifier = get_data_classifier(classifier_name)
    #To set the number of jobs to use for parallel processing
    data_classifier.n_jobs = -1
    #To reshape the data to a 2D array and scale it
    input_data = input_data.reshape(input_data.shape[0], -1)
    input_data = preprocessing.scale(input_data)

    #To perform 10-fold cross-validation to evaluate the classifier performance
    conf_mat_list, accuracy, precision, recall = cross_data_validation(input_data, input_labels, data_classifier)

    #To print the experiment results
    print(f" {data_type} data with {classifier_name} classifier:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("Confusion matrix:")
    print(conf_mat_list)


#To check if the number of command line arguments is valid
if len(sys.argv) != 4:
   print("Please provide the classifier name, data type, and data path as arguments when running Project1.py")
   #To exit the program if the number of command line arguments is not valid
   sys.exit()

#To set classifier_name, data_type, and data_path using command line arguments
classifier_name = sys.argv[1]
data_type = sys.argv[2]
data_dir = sys.argv[3]
#To run the experiment using the specified classifier, data type, and data path
run_prediction(classifier_name,data_type,data_dir)
