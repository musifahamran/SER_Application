# SER using a Flask Application
Used Flask to create a simple application to deploy SER models for prediction

Ensure that
* Python 3.7 is installed

### How to run

1. Install python libraries from requirements.txt
   * pip install -r requirements.txt
   * If issues with installing torchvision and torch libraries, either:
     * Install CUDA with the correct version, and follow this: https://varhowto.com/install-pytorch-1-6-0/
     #### Or
     * Use CPU verison with 
       * pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
2. Change project directories in main.py
3. To use different pretrained models, change the variable in main
4. Run the project

---------
#### Note: This application is an incomplete demo to test classificaiton of the SER models trained from https://github.com/musifahamran/FYP

---------
#### Screenshot of Demo Application:

![img.png](img.png)


