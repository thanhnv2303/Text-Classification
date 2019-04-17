import os
import settings


path = 'D:/PYTHON/train/'
#path = 'E:/Documents/PROJECT_2/Data/10_cate/train/'
list = os.listdir(path) 
numCate = len(list)
print('Có ',numCate,' thể loại:')
numDoc = 0 #tong so doc 
averageDoc = 0 # do dai trung binh cua 1 doc
for f in list:
    pathCate = path + f    #duong dan con 
    file = os.listdir(pathCate)  # liet ke cac van ban trong thu muc    
    lenDoc = len(file) # so luong doc trong 1 cate
    print(f,' có : ',len(file), 'văn bản.')
    numDoc += lenDoc 
    contenSize = 0
    for doc in file :
        pathDoc = pathCate +'/'+ doc      
        doc = open(pathDoc,errors='ignore')
        contentDoc = doc.read()    
        contenSize += len(contentDoc)
    contenSize = round(contenSize / lenDoc)   
    averageDoc += contenSize
print('=> Tổng số ',numDoc,' văn bản')
averageDoc = round(averageDoc / numCate )
print('=> Độ dài trung bình 1 văn bản là: ',averageDoc,'từ')