import unittest
from src.information_retriever.information_retriever import *


class TestIrBuild(unittest.TestCase):
    def setUp(self):
        self.video1 = {
              "header": "YouTube",
              "title": "Watched Get a Grip: How Long Can a MythBuster Hold a Grenade?",
              "titleUrl": "https://www.youtube.com/watch?v\u003dlrhwwK0vAPI",
              "subtitles": [{
                "name": "Discovery",
                "url": "https://www.youtube.com/channel/UCqOoboPm3uhY_YXhvhmL-WA"
              }],
              "time": "2019-09-14T01:05:11.424Z",
              "products": ["YouTube"],
              "activityControls": ["YouTube watch history"]
            }
        self.video2 = {
          "header": "YouTube",
          "title": "Watched Thank You Dark Souls",
          "titleUrl": "https://www.youtube.com/watch?v\u003dj0n3uhQqh3Q",
          "subtitles": [{
            "name": "TheSupremeSax12",
            "url": "https://www.youtube.com/channel/UCaj0PTSsnWo7vHDjUydNdWQ"
          }],
          "time": "2021-05-19T05:11:23.211Z",
          "products": ["YouTube"],
          "activityControls": ["YouTube watch history"]
        }
        self.video3 = {
          "header": "YouTube",
          "title": "Watched The toxic pit with a $3 admission fee",
          "titleUrl": "https://www.youtube.com/watch?v\u003dn-Ej2EtE744",
          "subtitles": [{
            "name": "Tom Scott",
            "url": "https://www.youtube.com/channel/UCBa659QWEk1AI4Tg--mrJ2A"
          }],
          "time": "2021-09-08T04:35:48.228Z",
          "products": ["YouTube"],
          "activityControls": ["YouTube watch history"]
        }

    def testBuild1(self):
        builtVideo1 = IrFeatureSet.ir_build(self.video1, 2019)
        self.assertEquals(builtVideo1.clas, 2019)
        self.assertEquals(IrFeature("Channel is Discovery", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains get", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains a", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains grip:", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains how", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains long", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains can", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains mythbuster", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains hold", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Title contains grenade?", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Month is 09", True) in builtVideo1.feat, True)
        self.assertEquals(IrFeature("Time is after 3pm", False) in builtVideo1.feat, True)

    def testBuild2(self):
        builtVideo2 = IrFeatureSet.ir_build(self.video2, 2021)
        self.assertEquals(builtVideo2.clas, 2021)
        self.assertEquals(IrFeature("Channel is TheSupremeSax12", True) in builtVideo2.feat, True)
        self.assertEquals(IrFeature("Title contains thank", True) in builtVideo2.feat, True)
        self.assertEquals(IrFeature("Title contains you", True) in builtVideo2.feat, True)
        self.assertEquals(IrFeature("Title contains dark", True) in builtVideo2.feat, True)
        self.assertEquals(IrFeature("Title contains souls", True) in builtVideo2.feat, True)
        self.assertEquals(IrFeature("Month is 05", True) in builtVideo2.feat, True)
        self.assertEquals(IrFeature("Time is after 3pm", False) in builtVideo2.feat, True)

    def testBuild3(self):
        builtVideo3 = IrFeatureSet.ir_build(self.video3, 2021)
        self.assertEquals(builtVideo3.clas, 2021)
        self.assertEquals(IrFeature("Channel is Tom Scott", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains the", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains toxic", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains pit", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains with", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains a", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains $3", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains admission", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Title contains fee", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Month is 09", True) in builtVideo3.feat, True)
        self.assertEquals(IrFeature("Time is after 3pm", False) in builtVideo3.feat, True)


class TestIrTrain(unittest.TestCase):
    def SetUp(self):
        self.video4 = {
              "header": "YouTube",
              "title": "Watched Goodbye Stranger (2010 Remastered)",
              "titleUrl": "https://www.youtube.com/watch?v\u003du8pVZ5hTGJQ",
              "subtitles": [{
                "name": "Supertramp - Topic",
                "url": "https://www.youtube.com/channel/UCpS0hJ21amjA5eBApEOiIFw"
              }],
              "time": "2020-04-10T00:58:32.816Z",
              "products": ["YouTube"],
              "activityControls": ["YouTube watch history"]
            }

    def trainTest1(self):
        classifier4 = IrClassifier.ir_train([IrFeatureSet.ir_build(self.video4)])
        prop4 = classifier4.get_proportion_dict()
        prob4 = classifier4.get_probability_dict()
        testProp4 = {

        }
        testProb4 = {
            "2017": 0,
            "2018": 0,
            "2019": 0,
            "2020": 0,
            "2021": 0,
            "2022": 0,
            "2023": 0
        }



class TestIrGamma(unittest.TestCase):
    def SetUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
