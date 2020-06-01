#!/bin/bash

# mkdir data

# Download and extract human activity data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip -d data/har
rm UCI\ HAR\ Dataset.zip

# Download and extract gesture data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00302/gesture_phase_dataset.zip
unzip gesture_phase_dataset.zip -d data/gesture
rm gesture_phase_dataset.zip

# Download and extract occupancy data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip
unzip occupancy_data.zip -d data/occupancy
rm occupancy_data.zip

# Download and extract traffic data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz
gunzip Metro_Interstate_Traffic_Volume.csv.gz
mkdir data/traffic
mv Metro_Interstate_Traffic_Volume.csv data/traffic/

# Download ozone dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data
mkdir data/ozone
mv eighthr.data data/ozone/

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
unzip household_power_consumption.zip -d data/power
rm household_power_consumption.zip

wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt
mkdir data/person
mv ConfLongDemo_JSI.txt data/person/