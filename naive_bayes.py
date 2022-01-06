# David Miller

import numpy as np
import sys
import os
import random

np.set_printoptions(threshold = sys.maxsize)

# triple is a tuple of 3 values (x, mean, std)
# the purpose of coding it this way is so i can zip
# the record in the test data with the means and standard deviations
# which are all numpy arrays for performance reasons
def gaussian(triple):
    return (1/(triple[2]*np.sqrt(2*np.pi)))*(np.e**(-1*(triple[0]-triple[1])*(triple[0]-triple[1])/(2*triple[2]*triple[2])))


# takes a record from the test data, a dictionary that gives the probability of a class C on key C
# and a dictionary that stores the mean and standard deviation for an attribute given a class C in the last two rows.
# p_class[C] = probability(Class = C)
# classes[C][-2][x] = mean of attribute x for class C
# classes[C][-1][x] = std of attribute x for class C

def argmax(record, p_class, classes):
    max_probability = 0
    prediction = []
    duplicate = 1
    p_x = 0
    p_x_c = {}
    for key in classes:
        # we exempt the last column because it is the class number and not an attribute
        p_x_c[key] = np.array(list(map(gaussian, zip(record[:-1], classes[key][-2][:-1], classes[key][-1][:-1])))) 
        #we calculate P(x) as we go since we need to use the sum rule
        #at each step we calculate P(x1|C)*P(x2|C)*...*P(xn|C)*P(C)
        p_x += np.prod(p_x_c[key])*p_class[key]

    for key in p_x_c:
        prob = np.prod(p_x_c[key])*p_class[key]/p_x
        if prob >= max_probability:
            if prob == max_probability:
                duplicate += 1
                prediction.append(key)
            else:
                duplicate = 1
                prediction = [key]
            max_probability = prob
    # Randomly choose between the classes tied for max probability,
    # if there's more than one class with the max probability
    prediction = random.choice(prediction)
    return (prediction, max_probability, duplicate)

def main():
    # main method
    if len(sys.argv) != 3:
        print("Usage: naive_bayes <training_data.txt> <test_data.txt>")
        exit(0)

    training_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    if not os.path.exists(training_data_path):
        print("Unable to find training file %s" % (training_data_path))
        exit(0)

    if not os.path.exists(test_data_path):
        print("Unable to find test file %s" % (test_data_path))
        exit(0)

    training_data_file = open(training_data_path)
    training_data = training_data_file.readlines()
    
    # classes is a dict that organizes the training records 
    # by class for calculating P(C) and P(x|C)
    # p_class is a dict that stores P(C == c) for a given key c
    classes = {}
    p_class = {}
    
    for data in training_data:
        record = np.array(list(map(float, data.split())), ndmin=2)
        if record[0][-1] in classes:
            classes[record[0][-1]] = np.append(classes[record[0][-1]], record, axis=0)
        else:
            classes[record[0][-1]] = record
    
    for key in sorted(classes.keys()):
        # Calculate P(C)
        p_class[key] = len(classes[key])/len(training_data)

        # Calculate the mean and std for each class and append them to the end of the matrix for easy access
        classes[key] = np.append(classes[key], np.array(list(map(np.mean, zip(*classes[key]))), ndmin=2), axis=0)
        # Need to use a lambda function so that the ddof argument can be set to 1, 
        # calculating sample standard deviation instead of population standard deviation
        classes[key] = np.append(classes[key], np.array(list(map(lambda x: np.std(x, ddof=1), zip(*classes[key]))), ndmin=2), axis=0)
        
        # Ensure no attribute has a std < 0.01
        for attribute in range(len(classes[key][0])-1):
            if classes[key][-1][attribute] < 0.01:
                classes[key][-1][attribute] = 0.01
            print("Class %d, attribute %d, mean =  %.2f, std =  %.2f" %(key, attribute+1, classes[key][-2][attribute], classes[key][-1][attribute] ))

    training_data_file.close()
    #print()

    test_data_file = open(test_data_path)
    test_data = test_data_file.readlines()
    classification_accuracy = 0
    item = 1
    for data in test_data:
        record = np.array(list(map(float, data.split())), ndmin=1)
        result = argmax(record, p_class, classes)
        accuracy = None
        if result[0] == record[-1]:
            accuracy = 1/result[2]
        else:
            accuracy = 0
        classification_accuracy += accuracy
        print("ID = %5d, predicted = %3d, probability = %.4f, true = %3d, accuracy = %4.2f" % (item, result[0], result[1], record[-1], accuracy))
        item += 1
    classification_accuracy /= len(test_data)
    test_data_file.close()
    #print()
    print("classification accuracy = %6.4f" % (classification_accuracy))



# run main method
if(__name__ == "__main__"):
    main()


