# Assigngment 5: Information Retrieval from Real Data
**Code Word:** 

**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Author Information
* **Name**: Davis Peterson
* **Email**: davpeterson@westmont.edu

## My Corpus:

For this project I chose to my YouTube watch history from my personal email, extracting it via Google Takeout. This came
in the form of a JSON containing 38,261 YouTube videos (and 1023 ads) that I watched from March 3, 2017, up until
December 2, 2023.

## Feature Selection:

While there was a very large collection of data to work with, Google doesn't track too much of the data of the actual
YouTube video. The features I chose to use 

# Interesting Test Cases

Test case 1: This video appears once in 2023, three times in 2021, and twice in 2020
- Title: Watched This Video Has 68,832,477 Views
- Channel: Tom Scott
- Hour watched: None
- Month: None
- **Results:** 2021 was predicted correctly everytime

Test case 2: This channel was watched predominately in 2017 and 2018, but not exclusively
- Title: None
- Channel: acdcVEVO
- Hour watched: None
- Month: None
- **Results:** 2017 was correctly predicted everytime, but 2018 was never outputted