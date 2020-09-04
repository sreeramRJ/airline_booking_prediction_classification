# our base image
FROM python:3

# install Python modules needed by the Python app
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

# copy files required for the app to run
COPY booking_prediction.py /

# run the application
CMD ["python", "./booking_prediction.py", "data/iata_1_1.csv", "data/events_1_1.csv.gz"]