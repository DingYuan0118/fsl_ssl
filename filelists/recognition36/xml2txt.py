import xml.etree.ElementTree as ET
import os
import sys

if __name__ == "__main__":
    xmls_path = "filelists\\recognition36\\boxlabels"
    target_path = "filelists\\recognition36\\boxlabels_txt"
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for xmlFilePath in os.listdir(xmls_path):
        print(os.path.join(xmls_path,xmlFilePath))
        try:
            tree = ET.parse(os.path.join(xmls_path,xmlFilePath))

            # 获得根节点
            root = tree.getroot()
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
            print("parse test.xml fail!")
            sys.exit()

        # objects = root.find("object")
        # print(len(objects))
        f = open(target_path +"/" + os.path.splitext(xmlFilePath)[0] + ".txt", 'w')
        # print(f)

 

        for bndbox in root.iter('bndbox'):
            node = []
            for child in bndbox:
                node.append(int(child.text))
            x1, y1 = node[0],node[1]
            x3, y3 = node[2],node[3]
            '''
            x2 ,y2 = x3 ,y1
            x4, y4 = x1, y3
            '''
            string = ''+str(x1)+','+str(y1)+','+str(x3)+','+str(y3);
            # print(string)
            f.write(string+'\n')

            
        f.close()