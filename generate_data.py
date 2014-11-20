import os
import random


num_train = int(raw_input('The number of training samples:'))
num_test = int(raw_input('The number of testing samples:'))
f_train = open("train.csv", "w")
f_test = open("test.csv", "w")
list_files = []
labels_training = set()
for path, subdirs, files in os.walk('dataset5'):
    for filename in files:
        fullname = os.path.join(path, filename)
        if "color" in fullname:
            label = fullname[ fullname.rindex('/')-1]
            labels_training.add(label)
            list_files.append([fullname,label])

random.shuffle(list_files)
for i in range(0, num_train):
    f_train.write(str(list_files[i][0]) + " " + str(list_files[i][1]) + os.linesep)
f_train.close()
for i in range(num_train, num_train+num_test):
    f_test.write(str(list_files[i][0]) + " " + str(list_files[i][1]) + os.linesep)
f_test.close()
print "Number of label in training set:" + str(len(labels_training))
