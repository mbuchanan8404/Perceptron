#  Matthew Buchanan
#  Perceptron Project
#  Intro to Neural Networks
#  Fall 2023


from csv import reader  # reader object reads a csv file line by line
from random import sample  # used to randomly sample a list without replacement
from random import shuffle
from Perceptron import Perceptron  # this is the Perceptron class in the Perceptron.py file


######################################################################
##### DATASET FUNCTIONS                                          #####
######################################################################

# Load the CSV file containing the inputs and desired output
#
#	dataset is a 2D matrix where each row contains 1 set of inputs plus the desired output
#		-for each row, columns 0-59 contain the inputs as floating point values
#		-column 60 contains the desired output as a character: 'R' for Rock or 'M' for Metal
#		-all values will be string values; conversion to appropriate types will be necessary
#		-no bias value is included in the data file
def load_csv(filename):
    # data will be the matrix containing the inputs
    data = list()
    # Standard Python code to read each line of text from the file as a row
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            # add current row to data
            data.append(row)
    return data


# Convert the input values in the specified column of the dataset from strings to floats
def convert_inputs_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip())


# Convert the desired output values, located in the specified column, to unique integers
# For 2 classes of outputs, 1 desired output will be 0, the other will be 1
def convert_desired_outputs_to_int(data, column):
    # Enumerate all the values in the specified column for each row
    class_values = [row[column] for row in data]
    # Create a set containing only the unique values
    unique = set(class_values)
    # Create a lookup table to map each unique value to an integer (either 0 or 1)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    # Replace the desired output string values with the corresponding integer values
    for row in data:
        row[column] = lookup[row[column]]
    return lookup


# Load the dataset from the CSV file specified by filename
def load_dataset(filename):
    # Read the data from the specified file
    data = load_csv(filename)
    # Convert all the input values from strings to floats
    for column in range(len(data[0]) - 1):
        convert_inputs_to_float(data, column)
    # Convert the desired outputs from strings to ints
    convert_desired_outputs_to_int(data, len(data[0]) - 1)
    return data

######################################################################
##### CREATE THE TRAINING SET                                    #####
######################################################################

# Create the training set
#	-Training set will consist of a random sampling without replacement of the data set
#
# Params:	data - the entire dataset
#
# Returns:	a matrix, or list of rows, containing only a subset of the input
#			vectors from the entire dataset
def create_training_set(data):
    return sample(data, 75)

######################################################################
##### CREATE A PERCEPTRON, TRAIN IT, AND TEST IT                 #####
######################################################################

# Step 1: Acquire the dataset
dataset = load_dataset('sonar_all-data.csv')

# Step 2: Create the training set
training_set = create_training_set(dataset)

# Step 3: Create the perceptron
bias = 1.0
weights = list(0.1 for i in range(60))
perceptron = Perceptron(bias, weights)

# Step 4: Train the perceptron
learning_rate = 0.01
number_of_epochs = 1500
perceptron.train(training_set, learning_rate, number_of_epochs)

# Step 5: Test the trained perceptron
predictions = perceptron.test(dataset)

# Step 6: Display the test results and accuracy of the perceptron
error_count = 0
for i in range(len(predictions)):
    # print('Actual: ' + str(predictions[i]) + '  Expected: ' + str(dataset[i][60]))
    if predictions[i] != dataset[i][60]:
        error_count += 1
error_rate = error_count / len(dataset)
print('\nAccuracy Rate: ' + str(round((1 - error_rate) * 100, 2)) + '%')
