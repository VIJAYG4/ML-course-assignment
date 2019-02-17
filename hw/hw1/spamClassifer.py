#author : vijay gandhi

#import stmts
from __future__ import division
import numpy as np
import scipy
import pandas as pd 


data = np.genfromtxt('spambasetrain.csv', delimiter=',')
#print(data.shape)
#print(type(myFile))

attributes = data[:,:-1]
labels     = data[:,-1]
#print(attributes.shape)
#print(labels.shape)

total_samples = len(labels)
nb_pos_class = np.count_nonzero(labels)
nb_neg_class = total_samples - nb_pos_class
#print(nb_pos_class)
#print(nb_neg_class)

pb_pos_class = nb_pos_class / total_samples
pb_neg_class = nb_neg_class / total_samples
#print(pb_pos_class)
print("Estimated value of P(C) for positive class = %.10f" %pb_pos_class)
print("Estimated value of P(C) for negative class = %.10f" %pb_neg_class)


#boolean indexing, create positive boolean mask
bool_mask = data ==1 
#print(bool_mask.shape, bool_mask[:5,-1])
pos_class_attributes = data[bool_mask[:,-1]]
pos_class_attributes = pos_class_attributes[:,:-1]
#print(pos_class_attributes.shape, pos_class_attributes[:5])

#boolean indexing, create negative boolean mask
bool_mask = data ==0
neg_class_attributes = data[bool_mask[:,-1]] 
neg_class_attributes = neg_class_attributes[:,:-1]
#print(neg_class_attributes.shape, neg_class_attributes[:5])

#validation (should print true)
#print(pos_class_attributes.shape[0] + neg_class_attributes.shape[0] == total_samples)


#calculate mean and standarad deviation of attributes for pos and neg classes separately
#mean for positive class
mean_pos_class_attributes = np.mean(pos_class_attributes, axis = 0)
#print(mean_pos_class_attributes.shape)

#std deviation for pos_class
#calculate std deviation according to the formula given
std_pos_class_attributes = np.divide(np.sum(np.square(pos_class_attributes - mean_pos_class_attributes) , axis =0), (nb_pos_class -1))
#print(std_pos_class_attributes.shape)

#zip mean and std as a tuple
zipped_mean_std_pos_class = zip(mean_pos_class_attributes,std_pos_class_attributes)
print("Estimate (mean,std) for positive class (9 pairs)")
print(zipped_mean_std_pos_class)

#mean for neg class
mean_neg_class_attributes = np.mean(neg_class_attributes, axis = 0)
#print(mean_neg_class_attributes.shape)

#std deviation for neg class
std_neg_class_attributes = np.divide(np.sum(np.square(neg_class_attributes - mean_neg_class_attributes) , axis =0), (nb_neg_class -1))
#print(std_neg_class_attributes.shape)

zipped_mean_std_neg_class = zip(mean_neg_class_attributes,std_neg_class_attributes)
print("Estimate (mean,std) for negative class (9 pairs)")
print(zipped_mean_std_neg_class)


#Read the test_data
test_data = np.genfromtxt('spambasetest.csv', delimiter=',')
actual_labels = test_data[:,-1]
#ignore the labels at the last column
test_data = test_data[:,:-1]
#print(test_data.shape)

nb_test_samples = test_data.shape[0]
 

def ml_sum_func(packed):
   mlf =[]
   for i in xrange(len(packed)):
    x, mean, variance = packed[i]
    ans = np.log(1 / np.sqrt(2* np.pi * variance)) +  -((x - mean)**2 / (2 * variance))
    mlf.append(ans)
   ml_sum = np.sum(mlf) 
   return ml_sum  




#check ln_exp_func
#print(ln_exp_func(0,0, 1/(2*np.pi)))

pred_labels = [] 
for i in xrange(nb_test_samples):
 #calculate MAP prob for pos class
	param_list = zip(list(test_data[i]), list(mean_pos_class_attributes), list(std_pos_class_attributes))
	pred_pb_pos_class = ml_sum_func(param_list) + np.log(pb_pos_class)
	#print(pred_pb_pos_class)

	param_list = zip(list(test_data[i]), list(mean_neg_class_attributes), list(std_neg_class_attributes))
	pred_pb_neg_class = ml_sum_func(param_list) + np.log(pb_neg_class)
	#print(pred_pb_neg_class)

	if pred_pb_pos_class >= pred_pb_neg_class:
		pred_labels.append(1)
	else:
		pred_labels.append(0)
#print(len(pred_labels))

print("Predicted classes for all test examples:")
print(pred_labels)

bool_mask = pred_labels==actual_labels
nb_correct = np.count_nonzero(bool_mask)
nb_incorrect = test_data.shape[0] - nb_correct

#print(nb_incorrect,nb_correct)
accuracy = np.mean(pred_labels == actual_labels)

print("Total examples classified correctly = %d" %nb_correct)
print("Total examples classified incorrectly = %d" %nb_incorrect)

percent_error = nb_incorrect / test_data.shape[0] * 100 
print("Percentage error on test examples = %f" %percent_error  )
#print(np.mean(pred_labels == actual_labels))

#######################################################################################################################
#implementing zero-R algorithm for comparison of accuracy
pred_labels = [0]* test_data.shape[0]
#accuracy of zero-R algorithm
#print(np.mean(pred_labels == actual_labels))