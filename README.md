# face_recognition_project

**The code can do the following**:


**a)** Recognize faces using pretrained model (haarcascade_frontalface.xml).

**b)** Compare the face on screen with the faces stored in folders and identify the face if found similar to one of the images stored in the folders. 

**c)** Create a .CSV file with the date in its name and mark attandance for the identified face.

**d)** If a new face is found, asks "Who is this?" and when the name is typed in the console, a new folder with the typed name is created and the new face on screen is saved.

**Python packages** we will need:
- openCV (cv2)
- face_recognition
- imutils
- pickle

Before starting to code:
- Create a folder named 'Images'.
- Inside the folder create more folders, one folder for each person that has to be identified. Name these folders after the person's name.
- Download haarcascade_frontalface.xml and save it in the same directory as the folder 'Images' and our code.
