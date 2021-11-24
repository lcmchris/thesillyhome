# thesillyhome
The silly home

This is a repo for routine creation for HomeAssistant.

Setup:
In this setup, we have a ec2 instance hosting our HomeAssistant mariaDB.

The stages of processing is as follows:

1) Data extraction
2) Learning model
3) HA routine creation

<h1> Data extraction </h1>
HomeAssistant stores state data. Extract and parse this data into a ML readable format

<h1> Learning model </h1>
As a phase one, our focus will be to predict lights using motion sensors, light sensors, other lights, device location, weather as inputs.

