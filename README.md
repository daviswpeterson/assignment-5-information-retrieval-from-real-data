# assignment-5-information-retrieval-from-real-data
This is the repository for Davis' assignment 5 work for CS-128: Information Retrieval and Big Data

# Things I could program

Okay so I have an idea for assignment 5 but I wanted to run it by you first to see if it could have your stamp of approval. Here's some basic info to start:

- Tool: text classifier of assignment 4
- File: my youtube watch history as a JSON

My goal: given a set of user inputs (keywords for title, channel name, time of day, etc.), what is the most likely year 
of Davis' life (2017-2023) that he could have watched a youtube video like this?

To use the premise of assignment 4 again, I understand I have to make an enhanced version of this text classifier. These
are plans towards improving it:

- More than just binary: originally, Connor and I had made our classifier purely on a binary data set so that the
effectiveness and the simplicity of the code would be easier to achieve. The classes for this new classifier, though,
are the years 2017-2023, so there are 7 different classes.
  - This will have implications for train(), gamma(), and present_features()
- User input: I plan on implementing user input in the code and allow for more varied inputs. The features being tested
on will be dependent upon what features the user deems important when initializing a new classifier.
  - One very key part of this code that I plan on implementing is allowing the user to choose which classes they want to
the classifier to be concerned with. This means the user can select n classes, and the rest of the methods will have to
be able to support that.
The implementation of build() will have to support variable amounts of types of features
- Adversarial engine: My runner will have to support the fact that there are meaningless videos in my watch history,
particularly ads.
- Maybe: Comprehensive .txt output: after a user is done using a classifier that they defined, it could be interesting
for the program to output a file presenting a lot of the statistics of the classifier, like a history of searches, average gamma, most common class, etc.

This about all I got. I would've brought this to you in person, but I'm fairly busy this week. If you think this isn't 
enough to be "enhanced" I'd love to add more to it because this would be a lot of fun to do! Thank!

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
- **Results:** 2017 was correctly predicted everytime, but not 2018 every