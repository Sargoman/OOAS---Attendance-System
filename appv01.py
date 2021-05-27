import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from flask import Flask, render_template, request


path = 'D:/Career/AI_ML Course via CareerEX/Project/AI Project/Behavior Analytics/code/ImagesAttendance/' 
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('D:/Career/AI_ML Course via CareerEX/Project/AI Project/Behavior Analytics/code/Attendance.csv', 'r+') as f:
        myDataList = f.read()
        nameList = ['unknown']
        # for line in myDataList:
        #     entry = line.split(',')
        #     nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
        #dtString = now.strftime('%H:%M:%S')
            dtString = now.isoformat(timespec='seconds') 
            f.writelines(f'\n{name},{dtString}')
            

encodeListKnown = findEncodings(images)
#print("Encoding complete")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/attendance', methods=['GET','POST'])
def attendance():
    if request.method == 'POST':


        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
    
    
            if True:
                imgs = cv2.resize(img,(0,0),None,0.25,0.25)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                facesCurFrame = face_recognition.face_locations(imgs)
                encodesCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)
    
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    #print(faceDis)
                    matchIndex = np.argmin(faceDis)
        
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        #print(name)
                        y1,x2,y2,x1 = faceLoc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2, y2),(0,255,0),2)
                        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                        markAttendance(name)
            
                cv2.imshow('webcam', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
    # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        return render_template("result.html")
          


if __name__ == '__main__':
    app.run(debug=True)





