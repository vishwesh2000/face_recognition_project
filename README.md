# face_recognition_project

**The code can do the following**:


**a)** Recognize faces using pretrained model (haarcascade_frontalface.xml).

**b)** Compare the face on screen with the faces stored in folders and identify the face if found similar to one of the images stored in the folders. 

**c)** Create a .CSV file with the date in its name and mark attandance for the identified face.

**d)** If a new face is found, asks "Who is this?" and when the name is typed in the console, a new folder with the typed name is created and the new face on screen is saved.

Python packages used:
- openCV (cv2)
- face_recognition
- imutils
- pickle

Before starting with the code:
- Create a folder named 'Images'.
- Inside the folder create more folders, one folder for each person that has to be identified. Name these folders after the person's name.
- Download haarcascade_frontalface.xml and save it in the same directory as the folder 'Images' and our code.

Few suggestions:
- Installation of face_recognition package is easier with Anaconda.
- Create a python virtual environment for this project and use Anaconda powershell(virenv) to run the code.
- Hardware implementation of the project could be done using RPI, however other developemental boards could be used based on requirement.
- A good database could also be created as part of the project and extend the number of people that are identifiable.
- Store 14 images of each person in their respective folders, make sure not all images are of the same illumination condition and angle. (For best results)
