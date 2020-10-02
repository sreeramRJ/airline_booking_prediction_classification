# Prediction of Users to book a travel - Data Science Challenge

## Goal

One of the main problems we face at travel is identifying users that will eventually book a trip to an advertised destination. In this challenge, you are tasked to build a classifier to predict the conversion likelihood of a user based on previous search events, with emphasis on the feature engineering and evaluation part.


## Data

You are provided with two sample data sets

- `events.csv.gz` - A sample of events collected from an online travel agency, containing:
  * `ts` - the timestamp of the event
  * `event_type` - either `search` for searches made on the site, or `book` for a conversion, e.g. the user books the flight
  * `user_id` - unique identifier of a user
  * `date_from` - desired start date of the journey
  * `date_to` - desired end date of the journey
  * `origin` - IATA airport code of the origin airport
  * `destination` - IATA airport code of the destination airport
  * `num_adults` - number of adults
  * `num_children` - number of children

- `iata.csv` - containing geo-coordinates of major airports
  * `iata_code` - IATA code of the airport
  * `lat` - latitude in floating point format
  * `lon` - longitude in floating point format


## Code

The code does the following:

- data preparation:
  - calculate the geographic distance between origins and destinations
  - convert raw data to a format suitable for the classification task
- experimental design:
  - split data into test and training sets in a meaningful way
- model:
  - a LGBMclassifier that predicts the conversion-likelihood of a user
  
Total runtime : ~ 8 min

## Execution instructions
* Install dependancies : pip install -r requirements.txt
* Run python executable(with input files as argument) : python src/booking_prediction.py data/iata_1_1.csv data/events_1_1.csv.gz

## Result
You can find the screenshot of result in result.png
