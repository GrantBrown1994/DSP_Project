# DSP_Project

This project is implemented in Python 3.11. To install the dependencies for this project and generate the reports:

1. Install Python 3.11.
2. Open a terminal window and navigate to the directory containing the project files. 
2. Create an virtual environment for the report/project. See [here](https://docs.python.org/3/library/venv.html) for more information.  
`> python -m venv .venv`
4. In the newly created environment, install the required packages from the requirements.txt file included with the project files.  
`(.venv) > pip install -r requirements.txt`
5. Open the report with Jupyter Notebooks.  
`(.venv) > jupyter notebook report.ipynb`  

7. To generate the .html version of the report,  
`(.venv) > jupyter nbconvert --to html report.ipynb`
