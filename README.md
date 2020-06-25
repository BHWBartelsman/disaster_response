# Disaster Response Pipeline Project

### Github repository

https://github.com/BHWBartelsman/disaster_response

### Background:
Following a disaster, there are a number of different problems that may arise. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, so predictive modeling can help classify different messages more efficiently.

In this project, I built an ETL pipeline that cleaned messages using NLTK. The text data was trained on a  classifier model using random forest. The final deliverable is Flask app that classifies input messages and shows visualizations of key statistics of the dataset.

### Files:
- process_data.py : ETL script to clean data into proper format by splitting up categories and making new  columns for each as target variables.
- train_classifier.py : Script to tokenize messages from clean data and create new columns through feature engineering. The data with new features are trained with a ML pipeline and pickled.
- run.py : Main file to run Flask app that classifies messages based on the model and shows data visualizations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
