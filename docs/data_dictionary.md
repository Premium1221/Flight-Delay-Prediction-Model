# Data Dictionary for Flight Delay Prediction Dataset (2023)

Snapshot of the core BTS On-Time fields used across the notebooks and app.
See `data/README.md` for sourcing instructions and `docs/README.md` for
additional documentation pointers.

| Feature                 | Type        | Description                                | Unit       | Range                          |
|:------------------------|:------------|:-------------------------------------------|:-----------|:-------------------------------|
| FL_DATE                 | Date        | Date of the flight                         | YYYY-MM-DD | 2023-01-01 to 2023-12-31       |
| AIRLINE                 | Categorical | Full name of airline                       | -          | Various US airline names       |
| AIRLINE_DOT             | Categorical | Airline name (with code) for DOT reporting | -          | Various                        |
| AIRLINE_CODE            | Categorical | IATA/ICAO airline code                     | -          | e.g., 'AA', 'DL', 'WN'         |
| DOT_CODE                | Numeric     | DOT-assigned airline code                  | -          | 100 to 1999                    |
| FL_NUMBER               | Numeric     | Flight number                              | -          | 1 to ~9999                     |
| ORIGIN_CITY             | Categorical | City of departure                          | -          | Major US cities                |
| DEST_CITY               | Categorical | City of arrival                            | -          | Major US cities                |
| ORIGIN                  | Categorical | Origin airport code                        | -          | 3-letter codes like 'JFK'      |
| DEST                    | Categorical | Destination airport code                   | -          | 3-letter codes like 'LAX'      |
| CRS_DEP_TIME            | Numeric     | Scheduled departure time                   | HHMM       | 0 to 2359                      |
| DEP_TIME                | Numeric     | Actual departure time                      | HHMM       | 0 to 2359                      |
| DEP_DELAY               | Numeric     | Departure delay                            | Minutes    | -60 to 1440+                   |
| TAXI_OUT                | Numeric     | Taxi-out time (before takeoff)             | Minutes    | 0 to ~100                      |
| WHEELS_OFF              | Numeric     | Time when plane took off                   | HHMM       | 0 to 2359                      |
| WHEELS_ON               | Numeric     | Time when plane landed                     | HHMM       | 0 to 2359                      |
| TAXI_IN                 | Numeric     | Taxi-in time (after landing)               | Minutes    | 0 to ~90                       |
| CRS_ARR_TIME            | Numeric     | Scheduled arrival time                     | HHMM       | 0 to 2359                      |
| ARR_TIME                | Numeric     | Actual arrival time                        | HHMM       | 0 to 2359                      |
| ARR_DELAY               | Numeric     | Arrival delay                              | Minutes    | -60 to 1000+                   |
| CANCELLED               | Binary      | 1 if cancelled, 0 otherwise                | 0 or 1     | 0 or 1                         |
| CANCELLATION_CODE       | Categorical | Reason for cancellation                    | -          | A = Carrier, B = Weather, etc. |
| DIVERTED                | Binary      | 1 if diverted, 0 otherwise                 | 0 or 1     | 0 or 1                         |
| CRS_ELAPSED_TIME        | Numeric     | Scheduled flight duration                  | Minutes    | 30 to 600+                     |
| ELAPSED_TIME            | Numeric     | Actual flight duration                     | Minutes    | 30 to 1000+                    |
| AIR_TIME                | Numeric     | Time spent in air                          | Minutes    | 10 to 900+                     |
| DISTANCE                | Numeric     | Distance flown                             | Miles      | 50 to 5000+                    |
| DELAY_DUE_CARRIER       | Numeric     | Carrier-related delay                      | Minutes    | 0 to 1000+                     |
| DELAY_DUE_WEATHER       | Numeric     | Weather-related delay                      | Minutes    | 0 to 1000+                     |
| DELAY_DUE_NAS           | Numeric     | NAS-related delay                          | Minutes    | 0 to 1000+                     |
| DELAY_DUE_SECURITY      | Numeric     | Security-related delay                     | Minutes    | 0 to 100+                      |
| DELAY_DUE_LATE_AIRCRAFT | Numeric     | Previous aircraft delay                    | Minutes    | 0 to 1000+                     |
| IS_DELAYED              | Binary      | 1 if ARR_DELAY > 15 mins                   | 0 or 1     | 0 or 1                         |
