import os

a = "g:\keras-transfer-learning-for-oxford102\data\jpg\image_00002.jpg".split(os.sep)[-2]
a = ["a", "b"]
b = [4, 5]
print(dict(zip(a, b)))