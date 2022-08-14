import random

# read everything in
lines = open("./Sorted/medvision_word_only_filter.txt").readlines()

# randomise order
random.shuffle(lines)

# split array up and write out according to desired proportions
open('./Sorted/medvision_resorted_train.txt', 'w').writelines(lines[100:])
open('./Sorted/medvision_resorted_test.txt', 'w').writelines(lines[:100])