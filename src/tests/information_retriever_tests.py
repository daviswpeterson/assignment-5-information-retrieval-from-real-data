import unittest
from src.information_retriever.information_retriever import *


class TestIrBuild(unittest.TestCase):
    def SetUp(self):
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

    def testBuild2(self):
        builtVideo2 = IrFeatureSet.ir_build(self.video2, 2021)
        self.assertEquals(builtVideo2.clas, 2021)

    def testBuild3(self):
        builtVideo3 = IrFeatureSet.ir_build(self.video3, 2021)
        self.assertEquals(builtVideo3.clas, 2021)


class TestIrTrain(unittest.TestCase):
    def SetUp(self):
        pass


class TestIrGamma(unittest.TestCase):
    def SetUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
