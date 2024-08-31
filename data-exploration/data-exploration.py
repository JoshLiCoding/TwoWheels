import cv2
import matplotlib.pyplot as plt
import os
import json

i = 0
list = list(os.walk('../dataset/labels'))
for f in list[0][2]:
    if i > 30:
        break
    print(f)
    i += 1

    with open('../dataset/labels/' + f) as file:
        jsonfile = json.load(file)

        image = cv2.imread('../dataset/images/' +jsonfile['imagename'])
        for bbox in jsonfile['children']:
            color = (255, 255, 255)
            if bbox['identity'] == 'cyclist':
                color = (0, 255, 0)
            elif bbox['identity'] == 'pedestrian':
                color = (0, 0, 255)
            cv2.rectangle(image, (bbox['mincol'], bbox['minrow']), (bbox['maxcol'], bbox['maxrow']), color, thickness=2)
            cv2.putText(image, bbox['identity'], (bbox['mincol'], bbox['minrow']), color=color, fontFace=0, fontScale=1)

        cv2.imwrite('images/'+str(i)+'.png', image)