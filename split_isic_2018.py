import csv
import os
import shutil
import random
import math
import random

csv_path=os.path.join('')   # path to the gt annotation csv, delete the first row(titles) before use
save_anno_path=''           # path to save the split 5 fold annotations(in 5 txt)
csv_reader = csv.reader(open(csv_path))

def split_2018_5fold():
    d = [[], [], [], [], [], [], []]
    count_list = [0, 0, 0, 0, 0, 0, 0]
    for line in csv_reader:
        #print(line)
        image_name = line[0] + '.jpg'
        if int(line[1])==1:
            disease_type = 0
            d[0].append([image_name,disease_type])
        elif int(line[2])==1:
            disease_type = 1
            d[1].append([image_name, disease_type])
        elif int(line[3])==1:
            disease_type = 2
            d[2].append([image_name, disease_type])
        elif int(line[4])==1:
            disease_type = 3
            d[3].append([image_name, disease_type])
        elif int(line[5])==1:
            disease_type = 4
            d[4].append([image_name, disease_type])
        elif int(line[6])==1:
            disease_type = 5
            d[5].append([image_name, disease_type])
        elif int(line[7])==1:
            disease_type= 6
            d[6].append([image_name, disease_type])
        count_list[disease_type]=count_list[disease_type]+1

    for i in range(7):
        random.shuffle(d[i])


    fold_list = [[],[],[],[],[]]

    for i in range(7):
        for j in range(5):
            split_len=len(d[i])/5
            #print(len(d[i]))
            #print(int(split_len * j), int(split_len * (j + 1)))
            for k in range(round(split_len*j),round(split_len*(j+1))):
                fold_list[j].append(d[i][k])
                #f.write(content[0]+' '+str(content[1])+'\n')
    total_sum=0
    for i in range(5):
        total_sum=total_sum+len(fold_list[i])
        print(len(fold_list[i]))
    #print(total_sum)
    for i in range(5):
        txt_file_path = os.path.join(save_anno_path,'ISIC_2018_fold', str(i) + '.txt')
        # print(txt_file_path)
        with open(txt_file_path, 'a') as f:
            for content in fold_list[i]:
                f.write(content[0] + ' ' + str(content[1]) + '\n')


split_2018_5fold()  # split the gt into 5 folds