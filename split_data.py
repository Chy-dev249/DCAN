import random
with open('/home/huiying/uda/BIWAA/data/Office31/webcam.txt', 'r') as f:
    lines = f.readlines()


cls_set = {}
cls_name = []
# 替换每一行中的字符
for i in range(len(lines)):
    path = lines[i].split(' ')[0]
    cls = lines[1].split(' ')[1]
    if cls not in cls_set:
        cls_set[cls] = []
    cls_set[cls].append(path)

new_lines = ''
for k,v in cls_set.items():
    temp = random.choice(cls_set[k])
    new_lines += temp + '' + k + '\n'

# 写入修改后的内容到新的文件
with open('/home/huiying/uda/BIWAA/data/Office31/webcam_.txt', 'w') as f:
    f.writelines(new_lines)