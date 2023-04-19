# # 打开文件并读取每一行
# with open('/home/huiying/uda/BIWAA/data/Office31/webcam.txt', 'r') as f:
#     lines = f.readlines()

# # 替换每一行中的字符
# for i in range(len(lines)):
#     lines[i] = lines[i].replace('webcam/images/', '/data/huiying/office31/webcam/')

# # 写入修改后的内容到新的文件
# with open('/home/huiying/uda/BIWAA/data/Office31/webcam_my.txt', 'w') as f:
#     f.writelines(lines)





# 写入修改后的内容到新的文件
# with open('/home/huiying/uda/DCAN/data/list/office_cida/a_10.txt', 'w') as f:
#     f.writelines(new_lines)













# 打开文件并读取每一行
with open('/home/huiying/uda/DCAN/data/list/office/dslr_31_my.txt', 'r') as f:
    lines = f.readlines()

new_lines = ''
# 替换每一行中的字符
for i in range(len(lines)):
    cls_num = lines[i].split(' ')[1].split('\n')[0]
    path = lines[i].split(' ')[0]

    if int(cls_num) < 10 or (int(cls_num) >= 21 and int(cls_num) <= 30):
        if int(cls_num) >= 21 and int(cls_num) <= 30:
            cls_num = int(cls_num) - 11
        temp = path + ' ' + str(cls_num) + '\n'
        
        new_lines += temp
        
        

# 写入修改后的内容到新的文件
with open('/home/huiying/uda/DCAN/data/list/office/dslr_20.txt', 'w') as f:
    f.writelines(new_lines)