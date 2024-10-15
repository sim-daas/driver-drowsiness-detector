# DRIVER DROWSINESS DETECTION

## ENVIRONMENT SETUP

Here's how you can create a basic Conda environment and install libraries using `pip` inside that environment:

### 1. **Install Conda** (if not already installed)
Make sure you have Conda installed (either through [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)).

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init zsh
```

### 2. **Create a Conda Environment**

Use the following command to create a new Conda environment:

```bash
conda create --name myenv python=3.11
```
- `myenv`: Name of your environment (you can change it).
- `python=3.11`: Python version you want to use in the environment.

### 3. **Activate the Environment**

Activate the newly created environment with the following command:

```bash
conda activate myenv
```

### 4. **Install Libraries using Pip**

To install packages using `pip` in your Conda environment, make sure the environment is activated, and then use the following command:

```bash
pip install -r requirements.txt
```

### 5. **Check Installed Packages**

You can verify the installed packages using:

```bash
pip list
```

---

### Index

1. **Objective and Need**  
2. **Proposal**  
3. **Data Flow Diagram**  
4. **Parts of the Solution**  
   - Camera Input  
   - OpenCV  
   - PyTorch Backend YOLO Object Detector on Custom Dataset  
   - Serial Communication  
   - Arduino + Buzzer  
5. **Hardware + Software Requirements**  
6. **Dataset (Training and Testing Information)**  
7. **Conclusion**  
8. **References**  

---

### Objective and Need

Driver drowsiness is a significant factor in road accidents, contributing to approximately 20% of all crashes worldwide. According to the National Highway Traffic Safety Administration (NHTSA), drowsy driving results in an estimated 100,000 police-reported crashes annually in the United States, leading to around 800 fatalities and 44,000 injuries. The need for effective driver monitoring systems is paramount to enhance road safety and reduce accident rates. By implementing machine learning techniques for driver drowsiness detection, we can identify fatigue levels in real-time, providing timely alerts to drivers and potentially preventing life-threatening accidents on the road.

### Proposal

This project aims to develop a **Driver Drowsiness Detection System** utilizing machine learning techniques to enhance road safety. The system will employ a camera to monitor the driver's facial features and eye movements in real-time, identifying signs of drowsiness through advanced algorithms.

1. **Data Acquisition**: The system will capture video input from a webcam, providing a continuous feed of the driver’s face.

2. **Drowsiness Detection**: We will implement a custom-trained YOLO (You Only Look Once) object detector model using PyTorch to analyze facial landmarks and eye states. This will allow for accurate detection of closed or drowsy eyes.

3. **Alert Mechanism**: Upon detecting drowsiness for a sustained period, the system will trigger an alert via serial communication to an Arduino connected to a buzzer, ensuring the driver receives immediate warnings.

4. **Prototype Development**: The final solution will be tested under real driving conditions to evaluate its effectiveness in preventing accidents caused by driver fatigue. 

### Data flow

By integrating these components, we aim to create a robust and reliable system that addresses the critical issue of driver drowsiness, ultimately contributing to safer roads and reducing accident rates.

              +------------------+
              |   Camera Input    |
              +--------+---------+
                       |
                       v
              +------------------+
              | Object Detection  |
              |  (CNN Algorithm)  |
              +--------+---------+
                       |
                       |  Detected
                       |
                       v
              +------------------+
              |  Serial Output    |
              +--------+---------+
                       |
                       v
              +------------------+
              | Arduino Serial    |
              |   Receive         |
              +--------+---------+
                       |
                       v
              +------------------+
              |   Buzzer Output   |
              +------------------+

## PARTS

Here’s a detailed breakdown for the different parts of our project

---

### Camera Input
The camera input serves as the primary sensory component of the drowsiness detection system. It captures real-time video footage of the driver’s face, providing continuous data for analysis. The choice of camera should support high-resolution video and operate effectively in varying lighting conditions to ensure accurate detection of eye states. The captured frames are then pre-processed to extract relevant features, such as the position of the face and eyes, which are critical for subsequent object detection.

---

### Object Detection (CNN Algorithm)
In this phase, we utilize a Convolutional Neural Network (CNN) for the object detection task, specifically trained to identify drowsy states by analyzing facial features. The architecture comprises several convolutional layers followed by pooling layers to reduce dimensionality and retain essential spatial features. The output of these layers is flattened and passed through fully connected layers to classify whether the driver’s eyes are open or closed.

The CNN is trained on a custom dataset, which includes a diverse set of images representing various states of eye openness and drowsiness, ensuring robust performance. Techniques such as data augmentation, dropout, and batch normalization are employed to improve the model’s generalization capabilities. The final output provides a bounding box around detected regions of interest (the eyes) and a confidence score indicating the likelihood of drowsiness.

---

### Serial Output
Upon detecting drowsiness, the system generates a serial output signal. This output is formatted according to the communication protocol established with the Arduino, ensuring reliable transmission of data. The serial output sends a predefined signal indicating the detection of drowsiness, which triggers the next phase of the system. 

This component relies on efficient data handling to minimize latency and ensure real-time performance. The system employs an asynchronous serial communication method to handle multiple operations simultaneously without blocking the main processing thread.

---

### Arduino Serial Receive
The Arduino microcontroller receives the serial output signal from the detection module. It is programmed to interpret the incoming data accurately and execute the appropriate response when a drowsiness condition is detected. The communication interface (usually UART) is configured to match the baud rate and format of the transmitted signal.

This part of the system is crucial for bridging the software and hardware components, enabling real-time interaction and responsiveness. The Arduino's ability to handle interrupts allows it to react promptly to the serial input, minimizing any delay in activating the buzzer.

---

### Buzzer Output
The buzzer output serves as an alert mechanism for the driver, activated by the Arduino upon receiving a drowsiness signal. The buzzer produces an audible sound, designed to quickly capture the driver's attention and mitigate potential hazards associated with drowsiness.

The output signal to the buzzer is managed through a digital pin on the Arduino, which can be programmed to control the duration and frequency of the sound emitted. This ensures that the alert is not only effective but can be tailored to the system's needs, such as adjusting volume or tone for optimal effectiveness in various environments.

--- 

