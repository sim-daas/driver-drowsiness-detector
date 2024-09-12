# DRIVER DROWSINESS DETECTION

### ENVIRONMENT SETUP

Here's how you can create a basic Conda environment and install libraries using `pip` inside that environment:

### 1. **Install Conda** (if not already installed)
Make sure you have Conda installed (either through [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)).

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
pip install requests matplotlib
```

### 5. **Check Installed Packages**

You can verify the installed packages using:

```bash
pip list
```
