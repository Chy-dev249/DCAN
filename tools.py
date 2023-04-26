import os
path_pre = '/home/huiying/uda/dcan/data/list/home/'
files = os.listdir(path_pre)
for txt in files:
    file_name = path_pre + txt
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 替换每一行中的字符
        for i in range(len(lines)):
            lines[i] = lines[i].replace('/data1/TL/data/office-home-65/', '/data/huiying/officehome/')

    # 写入修改后的内容到新的文件
    with open(file_name, 'w') as f:
        f.writelines(lines)

        


# # 打开文件并读取每一行
# with open('/home/huiying/uda/dcan/data/list/home/Art_65.txt', 'r') as f:
    

# p = '/home/huiying/uda/dcan/data/list/office_cida'


# import random
# with open('/home/huiying/uda/DCAN/data/list/office/amazon_31_my.txt', 'r') as f:
#     lines = f.readlines()

# new_lines = []
# cls_set = {}
# cls_name = []
# # 替换每一行中的字符
# for i in range(len(lines)):
#     class_name = lines[i].split(' ')[0].split('/')[5]
    

#     path = lines[i].split(' ')[0]
#     cls = lines[i].split(' ')[1].split('\n')[0]
#     cls_num = int(cls)
#     if cls_num not in cls_set:
#         cls_set[cls_num] = []
    

#     cls_set[cls_num].append(path)


# for k ,v in cls_set.items():
#     if k >=21:
#         new_lines.append(random.choice(cls_set[k]))

# for k in cls_set.keys():
#     print(k,len(cls_set[k])*0.05)
    
        



# 写入修改后的内容到新的文件
# with open('/home/huiying/uda/DCAN/data/list/office_cida/a_10.txt', 'w') as f:
#     f.writelines(new_lines)













# # 打开文件并读取每一行
# with open('/home/huiying/uda/DCAN/data/list/office/amazon_31_my.txt', 'r') as f:
#     lines = f.readlines()

# new_lines = ''
# # 替换每一行中的字符
# for i in range(len(lines)):
#     cls_num = lines[i].split(' ')[1]

#     if int(cls_num) < 10 or (int(cls_num) >= 21 and int(cls_num) <= 30):
#         new_lines += lines[i]
        
        

# # 写入修改后的内容到新的文件
# with open('/home/huiying/uda/DCAN/data/list/office/amazon_20.txt', 'w') as f:
#     f.writelines(new_lines)