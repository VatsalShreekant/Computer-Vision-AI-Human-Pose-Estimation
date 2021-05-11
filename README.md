# Computer-Vision-AI-Human-Pose-Estimation

Link to website: <a href="https://vatsalshreekant.github.io/Computer-Vision-AI-Human-Pose-Estimation/index.html" target="_blank" title="AA06">AA06</a> 

## Objective: 
The objective was to apply computer vision techniques based on Artificial Intelligence  for human pose detection that gives real-time feedback. The system assesses a yoga pose being performed and gives real time feedback in the form of error indicators along with a score to measure the accuracy of the pose being performed.

## Motivation:
Recent advancements in Deep Learning has added a huge boost to the already rapidly developing field of computer vision. Pose estimation is a subset of computer vision that refers to computer vision techniques that detect human figures in images and video, so that one could determine, for example, where someone’s ankle shows up in an image. The motivation behind this project was to detect movement and posture errors in performing yoga poses. The idea was to build an effective yoga programme for each user to help them improve their posture and health.

## Approach:
Building a human pose detection system with the PoseNet model entailed dividing the process into 3 steps: Data Collection, Model Training and Model Testing. 
<kbd>
![Capture](https://user-images.githubusercontent.com/32462270/117875952-c5d3b780-b270-11eb-8ff0-0ff24d2180e9.PNG)
</kbd>
<kbd>
![Capture](https://user-images.githubusercontent.com/32462270/117876193-12b78e00-b271-11eb-907d-939103d29c03.PNG)
</kbd>

<ins>Data Collection</ins>: Using PoseNet, when a pose would be detected, the 14-(x,y) inputs and target data would be fed into the neural network defined object. A setTimeout() function was used to record the poses for 5 seconds.
  
<ins>Model Training</ins>: The existing data would be loaded into the JavaScript environment. Once the downloaded json file was attached to the model training file, the neural network was then trained across 50 epochs.
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117877283-5f4f9900-b272-11eb-8116-76804aac5d36.PNG">
</kbd>
</p>
<ins>Model Testing</ins>: The model was trained by determining a better way to collect the angles based on specified points as well as a scoring system. For the scoring system, two comparisons were run. Firstly, the model compared the user’s chosen pose with the pose being performed on camera. The second comparison was dependant upon the first. The model then compared each incoming angle against some target angle determined as the average of all individual angles computed for each pose.

## Performance Results:
The performance of the model is quite promising. For poses such as ‘Mountain’ and ‘Warrior 2’, the results, when performed correctly, classifies the current pose. 
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117877564-b6ee0480-b272-11eb-93fd-8fb7f19078b1.PNG">
</kbd>
</p>

Upon evaluation of the model, the model is able to classify the poses with an accuracy of 82.9%. Incorrect poses can be detected immediately, and outputs real-time feedback to the user with error indicators.
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117877733-e7ce3980-b272-11eb-98da-895b47221104.PNG">
</kbd>
</p>

The performance of the model drops when the detection of keypoints fails. The scenarios include dark background or environment, as well as multiple users being in the frame of recording. The pose estimation uses a single-pose estimation model to perform data points collection, before being fed to the model classifier.

Attempt of performing ‘Tree’ pose with some computed angles falling below a threshold of 10° and 20° error:
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117879304-cd955b00-b274-11eb-8648-b99d2364f4d3.PNG">
</kbd>
</p>

Attempt of performing ‘Goddess’ pose with some computed angles falling below margin of 20° error:
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117879399-ef8edd80-b274-11eb-9302-92032ace12a7.PNG">
</kbd>
</p>

Attempt of performing ‘Goddess’ pose with some computed angles falling below a threshold of 10° and 20° error:
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117879448-02091700-b275-11eb-9225-76db4fe4fe99.PNG">
</kbd>
</p>

Attempt of performing ‘Warrior 2’ pose with some computed angles falling below a threshold of 10° and 20° error:
<p align="center">
<kbd>
<img src="https://user-images.githubusercontent.com/32462270/117879514-15b47d80-b275-11eb-8f1c-aaa991801432.PNG">
</kbd>
</p>




