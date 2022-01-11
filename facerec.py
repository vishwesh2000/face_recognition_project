# Extracting features from face
from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
#Get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()

# Face recognition on LIVE WEBCAM FEED
import face_recognition
import pickle
import cv2
import os
import _datetime
import time
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
print("Streaming started")
video_capture = cv2.VideoCapture(0)# + cv2.CAP_DSHOW) 
def attendance(name):
    moment=time.strftime("%Y-%b-%d",time.localtime())
    if os.path.exists('Attend'+moment+'.csv'):
        with open('Attend'+moment+'.csv','r+',newline="\n") as f:
            DataList = f.readlines()
            knownNames = []
            for data in DataList:
                ent = data.split(',')
                knownNames.append(ent[0])
        with open('Attend'+moment+'.csv','a',newline="\n") as f:
            if name not in knownNames:
                curr=_datetime.date.today()
                dt=curr.strftime('%d%m%Y_%H:%M:%S')
                f.writelines(f'\n{name}, {dt}, P')
    else:
        open('Attend'+moment+'.csv','w')
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=3,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase    
    # we have multiple embeddings for multiple fcaes
    img_counter = 0
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        #set name =unknown if no encoding matches
        name = "Unknown"
        
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                
            #set name which has highest count
            name = max(counts, key=counts.get)
        else: # To store the unknown new face with name
            new_name = input("Who is this?")
            path_2 = os.path.join('Images',new_name) 
            for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255),
                    thickness = 2)
                    #cv2.putText(frame, new_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    #0.75, (0, 255, 0), 2)
                    sub_face = frame[y:y+h, x:x+w]
                    FaceFileName = new_name + str(y+x) + ".jpg"
            if not os.path.exists(path_2):
                key = cv2.waitKey(1) & 0xFF
                os.mkdir(path_2)
                print("Directory '% s' created" % new_name)
                cv2.imwrite(os.path.join(path_2,FaceFileName),sub_face)
            else:
                cv2.imwrite(os.path.join(path_2,FaceFileName),sub_face)


                
            cv2.imshow("Frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            attendance(name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)              
    cv2.imshow("Frame", frame)
    path_3 = os.path.join('Images',name)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = name + str(y+x) + ".jpg".format(img_counter)
        cv2.imwrite(os.path.join(path_3, img_name),frame)
        
        print("Image written!".format(img_name))
        img_counter += 1
video_capture.release()
cv2.destroyAllWindows()


# Face recognition in IMAGES
import face_recognition
import pickle
import cv2
import os
 
#find path of xml file containing haarcascade file
cascPathface = os.path.dirname(
 cv2.__file__) + "haarcascade_frontalface_default.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
#Find path to the image you want to detect face and pass it here
image = cv2.imread(Path-to-img)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#convert image to Greyscale for haarcascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
 
# the facial embeddings for face in input
encodings = face_recognition.face_encodings(rgb)
names = []
# loop over the facial embeddings incase
# we have multiple embeddings for multiple fcaes
for encoding in encodings:
    #Compare encodings with encodings in data["encodings"]
    #Matches contain array with boolean values and True for the embeddings it matches closely
    #and False for rest
    matches = face_recognition.compare_faces(data["encodings"],
    encoding)
    #set name =inknown if no encoding matches
    name = "Unknown"
    # check to see if we have found a match
    if True in matches:
        #Find positions at which we get True and store them
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            #Check the names at respective indexes we stored in matchedIdxs
            name = data["names"][i]
            #increase count for the name we got
            counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
    
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    else: # To store the unknown new face with name
        faces = faceCascade.detectMultiScale(gray,
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(60, 60),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    cv2.imshow("Frame", image)
    cv2.waitKey(0)
