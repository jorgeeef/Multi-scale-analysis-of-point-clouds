How to clone this project:
============================================================================================

# sudo apt install git


# STEP 1 - open cmd in the folder where you need keep the project and clone the GitHub repo: 
git clone https://github.com/jorgeeef/Multi-scale-analysis-of-point-clouds.git

# STEP 2 - enter the project folder
cd Multi-scale-analysis-of-point-clouds

# STEP 3 - open it in VS Code
code .


#sudo apt update
#sudo apt install python3.12-venv


# STEP 4 - recreate environment 
python -m venv venv

# STEP 5 - activate the environment 
Linux: source venv/bin/activate
Windows: .\venv\Scripts\Activate.ps1

# STEP 6 - install dependencies
pip install -r requirements.txt

# STEP 7 - create data folder and add the obj files you need to run
mkdir data
# Then add the obj files here

# STEP 8 - how to run the code
python main.py

============================================================================================
# To get updates
git pull


