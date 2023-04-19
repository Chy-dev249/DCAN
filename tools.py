# # 打开文件并读取每一行
# with open('/home/huiying/uda/BIWAA/data/Office31/webcam.txt', 'r') as f:
#     lines = f.readlines()

# # 替换每一行中的字符
# for i in range(len(lines)):
#     lines[i] = lines[i].replace('webcam/images/', '/data/huiying/office31/webcam/')

# # 写入修改后的内容到新的文件
# with open('/home/huiying/uda/BIWAA/data/Office31/webcam_my.txt', 'w') as f:
#     f.writelines(lines)


import random
with open('/home/huiying/uda/DCAN/data/list/office/amazon_31_my.txt', 'r') as f:
    lines = f.readlines()

new_lines = []
cls_set = {}
cls_name = []
# 替换每一行中的字符
for i in range(len(lines)):
    class_name = lines[i].split(' ')[0].split('/')[5]
    

    path = lines[i].split(' ')[0]
    cls = lines[i].split(' ')[1].split('\n')[0]
    cls_num = int(cls)
    if cls_num not in cls_set:
        cls_set[cls_num] = []
    

    cls_set[cls_num].append(path)


# for k ,v in cls_set.items():
#     if k >=21:
#         new_lines.append(random.choice(cls_set[k]))

for k in cls_set.keys():
    print(k,len(cls_set[k])*0.05)
    
        



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