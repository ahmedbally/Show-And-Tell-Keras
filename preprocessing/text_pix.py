import os
import numpy as np

path = "../datasets/Flickr8k_text/"
extension = "gui"

docs = []
images = []
for file in os.listdir(path):
    if str(file)[-3:] == extension:
        with open(path + str(file), 'r') as f:
            docs.append((str(file)[:-3] + "jpg#0	"+f.read().replace("\n"," ").replace(",","").replace("\t","")))
            #docs.append(str(file)[:-3] + "jpg")
            #docs.append(f.read())
            #mages.append(str(file)[:-3] + "jpg")



'''docs = list(map(lambda x: x.replace("\n", " "), docs))
docs = list(map(lambda x: x.replace(",", ""), docs))
docs = list(map(lambda x: x.replace("\t", ""), docs))
docs = list(map(lambda x:images[docs.index(x)]+"#0	" + x , docs ))
docs = list(map(lambda x:x, images ))'''

f=open('../datasets/Flickr8k_text/test.txt','w+')
f.write("\n".join(docs))
f.close()