# MLP Assignment

## Overview
This project involves implementing a Multi-Layer Perceptron (MLP) from scratch to perform various tasks, including XOR classification, sinusoidal regression, and letter recognition.

## Features
- MLP with customizable input, hidden, and output layers
- Backpropagation learning algorithm
- Training and evaluation on multiple datasets

## Installation

```bash
git clone git@github.com:Lando-00/Connectionist_Project.git
pip install -r requirements.txt
```

## File Structure
1. mlp/ Directory:

	- This folder will store your custom MLP class and related code. It keeps your implementation modular and reusable.

	- File: mlp.py: Contains the MLP class definition.

	- File: __init__.py: Marks the folder as a Python package. It can remain empty.

2. main.py:

	- The entry point of your project. You’ll import the MLP class here and write training, evaluation, and testing logic.
3. requirements.txt:

	- Lists project dependencies. Since you’re using Conda, this file is optional unless you plan to share or deploy your project.
4. .gitignore:

	- Prevents unnecessary or sensitive files (e.g., __pycache__/, *.pyc, *.env) from being tracked by Git.
5. README.md:

	- Documents the project (e.g., purpose, installation steps, usage).
