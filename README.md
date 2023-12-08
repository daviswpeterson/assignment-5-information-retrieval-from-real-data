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
YouTube video. The features I chose to use were keywords in the title of the video, the name of the YouTube channel
associated with the video, the month the video was watched, and whether the video was watched after 3pm. The rest of the
usable information was fairly generic, like the second or millisecond the video was watch at, the link to the YouTube
video, some values that were equal across all videos.

## Implementation decisions

There were two main "enhancements" to the original text classifier code that Connor and I built that utilized. First of
all, I've divided the new corpus into 7 different classes, which makes the code more complicated. Secondly, I added user
input to streamline the classifying process and give them more freedom. The following is how this altered my strategy:

### build()

The build method is actually very similar to the one used in the previous assignment. The ony alteration is that instead
of accepting a string argument, it now takes in a dictionary representation of a YouTube video from the JSON.

**How to use:** All it requires is a dictionary with standard YouTube key-value pairs. This is what a YouTube video's
dictionary description looks like:

```
{
  "header": "YouTube",
  "title": "Watched Hill Hill Hill Hill, debunked, debunked",
  "titleUrl": "https://www.youtube.com/watch?v\u003dNUyXiiIGDTo",
  "subtitles": [{
    "name": "Tom Scott",
    "url": "https://www.youtube.com/channel/UCBa659QWEk1AI4Tg--mrJ2A"
  }],
  "time": "2021-09-06T16:32:05.615Z",
  "products": ["YouTube"],
  "activityControls": ["YouTube watch history"]
}
```

The only keys that are used and analyzed by this program are title, name, and time. These are parsed in build.

### train()

I drastically changed train as compared to the previous assignment. Since the last classifier only had two classes, this
allowed us (Connor and I) to make some inefficient design choices such as unnecessary if statements. I was able to cut
down on this code via dictionaries, which I made ample use of in the project. So while there are five additional classes,
the code is actually much simpler.

**How to use:** 

### Classifier constructor

I added a new argument to the constructor which is a variable called *proportion_dict*. What this stores is how much of
each class appeared in the training set. With the Tweet Classifier, both classes had an equal amount of items, but now
since each class (year of my life) have different amounts of YouTube videos watched, I'll have to account for that in
gamma. This dictionary is computed in train and used in gamma.

**How to use:** 

### gamma()

While gamma is very similar to its original model, some extra data was needed to make precise calculations. Instead of
assuming that the distributions of classes was equal, I brought in the exact amount of each class into gamma via a
dictionary. 

**How to use:** 

### runner

Runner has been completely revamped. The original code has been completely altered via user input. This had the positive
side effect of making testing of the accuracy much easier. I had to add a method to support turning user inputs into
dictionaries.

**How to use:** 