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
trains=[]
tests=[]
train=len(docs)*.8
i=0
for file in os.listdir(path):
    if str(file)[-3:] == extension:
        with open(path + str(file), 'r') as f:
            if i < train:
                trains.append((str(file)[:-3] + "jpg"))
            else:
                tests.append((str(file)[:-3] + "jpg"))
        i+=1

'''docs = list(map(lambda x: x.replace("\n", " "), docs))
docs = list(map(lambda x: x.replace(",", ""), docs))
docs = list(map(lambda x: x.replace("\t", ""), docs))
docs = list(map(lambda x:images[docs.index(x)]+"#0	" + x , docs ))
docs = list(map(lambda x:x, images ))'''

f=open('../datasets/Flickr8k_text/Flickr_8k.token.txt','w+')
f.write("\n".join(docs))
f.close()

f=open('../datasets/Flickr8k_text/Flickr_8k.trainImages.txt','w+')
f.write("\n".join(trains))
f.close()

f=open('../datasets/Flickr8k_text/Flickr_8k.devImages.txt','w+')
f.write("\n".join(tests))
f.close()