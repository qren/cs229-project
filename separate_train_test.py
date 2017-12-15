import random

name = 'flightdata'

lines = open('{}.txt'.format(name)).readlines()
random.shuffle(lines)
training_num = int(len(lines)*0.7)
training = lines[0:training_num]
testing = lines[training_num:]
open('{}_training.txt'.format(name), 'w').writelines(training)
open('{}_testing.txt'.format(name), 'w').writelines(testing)