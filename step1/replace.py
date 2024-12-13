# 传入文件(file),将旧内容(old_content)替换为新内容(new_content)
def replace(file, old_content, new_content):
    content = read_file(file)
    content = content.replace(old_content, new_content)
    rewrite_file(file, content)

# 读文件内容
def read_file(file):
    with open(file, encoding='UTF-8') as f:
        read_all = f.read()
        f.close()

    return read_all

# 写内容到文件
def rewrite_file(file, data):
    with open(file, 'w', encoding='UTF-8') as f:
        f.write(data)
        f.close()

# 替换操作(将test.txt文件中的'Hello World!'替换为'Hello Qt!')
replace(r"/share1/home/zhangjiarui/Projects/tip-NLOS-LTM/step1/datasets/MNIST/train/sharp_1.txt", 'NLOS-Passive', 'NLOS_Passive')
replace(r"/share1/home/zhangjiarui/Projects/tip-NLOS-LTM/step1/datasets/MNIST/val/sharp_1.txt", 'NLOS-Passive', 'NLOS_Passive')
replace(r"/share1/home/zhangjiarui/Projects/tip-NLOS-LTM/step1/datasets/MNIST/test/sharp_1.txt", 'NLOS-Passive', 'NLOS_Passive')



