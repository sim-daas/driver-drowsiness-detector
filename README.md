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
