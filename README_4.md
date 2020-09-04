# travel audience Data Science Challenge

## Goal

One of the main problems we face at travel audience is identifying users that will eventually book a trip to an advertised destination. In this challenge, you are tasked to build a classifier to predict the conversion likelihood of a user based on previous search events, with emphasis on the feature engineering and evaluation part.


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


## Tasks

Your code needs to do the following:

- data preparation:
  - calculate the geographic distance between origins and destinations
  - convert raw data to a format suitable for the classification task
- feature_engineering:
  - based on the given input data, compute and justify three features of your choice that are relevant for predicting converters
- experimental design:
  - split data into test and training sets in a meaningful way
- model:
  - a classifier of your choice that predicts the conversion-likelihood of a user

Use your best judgment to define rules and logic to compute each feature. Don't forget to comment your code!


## Deliverables

Code & comments that satisfy the tasks and demonstrate your coding style in Python or R. In addition, instructions on how to run your code.

We'll be evaluating the quality of your code, communication, and general solution design. We won't evaluate the actual performance of your model.
