import os

path = os.getcwd()
filenames = os.listdir(path)

i = 0;

for filename in filenames:
    os.rename(filename, "go"+str(i)+".jpg")
    i += 1
    