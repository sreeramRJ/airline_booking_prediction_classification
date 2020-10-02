# our base image
FROM python:3

# install Python modules needed by the Python app
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

# copy files required for the app to run
COPY src/booking_prediction.py /
COPY data/iata_1_1.csv /
COPY data/events_1_1.csv.gz /

# run the application
CMD ["python", "booking_prediction.py", "iata_1_1.csv", "events_1_1.csv.gz"]