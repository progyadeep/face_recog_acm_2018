# WHAT IS THIS
This is a program written in python using the <a href="https://pypi.org/project/face_recognition/" target="_blank">face_recognition</a> module. It can be <b>trained using a set of known images</b> and then the program will recognize the supplied image accordingly.

# HOW IT WORKS
A detailed documentation is available in the file <a href="https://github.com/progyadeep/face_recog_acm_2018/blob/master/DOCUMENTATION.md">DOCUMENTATION.md</a>.<br/>
Here's a brief overview of the working.

<ul>
<li> The folder <b>known_faces</b> contains all the known faces. Inside this folder, there are <b>n subfolders</b> named after the label of the faces they contain</li>
<li> The files <b>known_encodings.txt</b> and <b>known_names.txt</b> contain the information retrieved from the known faces, and their respective labels, respectively. This information is generated by the file <b>trainer.py</b> and will be needed later to recognize a given image.</li>
<li> The file <b>trainer.py</b>, as the name suggests, is the trainer. it extracts face encodings from the known images and places them in the <b>known_encodings.txt</b> file, and their labels in the <b>known_names.txt</b>.</li>
<li> The file <b>recognize.py</b> uses the data generated by the trainer to recognize a given image.</li>
</ul>
  
  # DEPENDENCIES
  <ul>
  <li> Linux or MacOS. Results are not guaranteed to be correct on Windows.</li>
  <li> Python 3
    <ul>
      <li> numpy</li>
      <li> cv2</li>
      <li> dlib</li>
      <li> face_recognition</li>
    </ul></li>
  </ul>
  The above python library dependences can be installed directly using the command:  
  <br/><br/>  
  > pip install xyz
  
  where xyz is the name of the dependancy as listed above. <b>Make sure to have the latest version of pip and setuptools.</b>
  
  # HOW TO RUN IT
  With the dependencies resolved, place all the known images inside the <b>known_faces</b> folder as follows: All images for Steve Jobs should be placed under <b>known_faces/Steve Jobs/</b>. One subfolder for each person. The images will automatically be labelled by the subfolder name.  
  
Then, run  

  > python trainer.py
  <br/>
  Then run  
  <br/>
  
  > python recognize.py
  <br/>
  The code will ask you for the path of the test image, which is the image that you want to recognize the face in. Enter the path of the image with respect to the current directory.
