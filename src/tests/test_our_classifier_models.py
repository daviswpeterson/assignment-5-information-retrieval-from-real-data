"""
Unit tests for my functions in `classifier.our_classifier_models`.
"""

import unittest
from src.classifier.our_classifier_models import *
from nltk.corpus import twitter_samples
import io
import sys

__author__ = "Connor Rogstad, Davis Peterson"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Connor Rogstad", "Davis Peterson"]
__license__ = "MIT"
__email__ = ["crogstad@westmont.edu", "davpeterson@westmont.edu"]


class TweetFeatureSetTest(unittest.TestCase):

    def setUp(self):
        json_content = ["Positive tweet positive", "this bad :((( sad"]
        # THE CONTENT OF THE JSON WILL BE SPLIT IN OUR RUNNER AND ONLY SINGLE TWEETS GET PASSED INTO BUILD
        tweet1 = json_content[0]
        tweet2 = json_content[1]
        # should get rid of the :((( and instead leave the :(

        self.actual_tweet_feature_set1 = TweetFeatureSet.build(tweet1, "positive")
        self.actual_tweet_feature_set2 = TweetFeatureSet.build(tweet2, "negative")

        # Tweet Contains Features
        self.feature1 = TweetFeature("Contains \"positive\"", True)
        self.feature2 = TweetFeature("Contains \"tweet\"", True)

        self.feature3 = TweetFeature("Contains \"this\"", True)
        self.feature4 = TweetFeature("Contains \"bad\"", True)
        self.feature5 = TweetFeature("Contains \":(((\"", True)
        self.feature6 = TweetFeature("Contains \"sad\"", True)
        # Tweet Contains FeatureSet
        self.contains_feature_set1 = TweetFeatureSet({self.feature1, self.feature2}, "positive")
        self.contains_feature_set2 = TweetFeatureSet({self.feature3, self.feature4, self.feature5, self.feature6},
                                                     "negative")

        # Tweet Length(in chars) Features
        self.feature7 = TweetFeature("Amount of characters", 23)
        self.feature8 = TweetFeature("Amount of characters", 17)
        # Tweet Length(in chars) FeatureSet
        self.length_feature_set1 = TweetFeatureSet({self.feature7}, "positive")
        self.length_feature_set2 = TweetFeatureSet({self.feature8}, "negative")

        # Tweet Count Features
        self.feature9 = TweetFeature("Amount of \"positive\"", 2)
        self.feature10 = TweetFeature("Amount of \"tweet\"", 1)
        self.feature11 = TweetFeature("Amount of \"(\"", 0)
        self.feature12 = TweetFeature("Amount of \")\"", 0)

        self.feature13 = TweetFeature("Amount of \"this\"", 1)
        self.feature14 = TweetFeature("Amount of \"bad\"", 1)
        self.feature15 = TweetFeature("Amount of \"(\"", 3)
        self.feature16 = TweetFeature("Amount of \")\"", 0)
        self.feature17 = TweetFeature("Amount of \":(((\"", 1)
        self.feature18 = TweetFeature("Amount of \"sad\"", 1)
        # Tweet Count FeatureSet
        self.count_feature_set1 = TweetFeatureSet({self.feature9, self.feature10, self.feature11, self.feature12}, "positive")
        self.count_feature_set2 = TweetFeatureSet({self.feature13, self.feature14,
                                                   self.feature15, self.feature16, self.feature17, self.feature18}, "negative")

    def test_build_contains(self):
        self.assertIn(self.feature1, self.actual_tweet_feature_set1.feat)
        self.assertIn(self.feature2, self.actual_tweet_feature_set1.feat)
        self.assertIn(self.feature4, self.actual_tweet_feature_set2.feat)
        self.assertIn(self.feature5, self.actual_tweet_feature_set2.feat)

    def test_build_length(self):
        self.assertIn(self.feature7, self.actual_tweet_feature_set1.feat)
        self.assertIn(self.feature8, self.actual_tweet_feature_set2.feat)

    def test_build_count(self):
        self.assertIn(self.feature9, self.actual_tweet_feature_set1.feat)
        # self.assertIn(self.feature11, self.actual_tweet_feature_set1.feat)
        self.assertIn(self.feature13, self.actual_tweet_feature_set2.feat)
        self.assertIn(self.feature14, self.actual_tweet_feature_set2.feat)


class TweetClassifierTest(unittest.TestCase):

    def setUp(self):
        self.M0_file_contents = ["0 0 0 1", "0 0 1 0", "1 0 0 0", "0 0 0 0"]
        self.ME1_file_contents = ["0 1 1 1", "1 0 0 1", "1 1 1 0", "1 1 0 0"]

        self.feat1 = Feature("First word is 1", True)
        self.feat2 = Feature("Second word is 1", True)
        self.feat3 = Feature("Third word is 1", True)
        self.feat4 = Feature("Last word is 1", True)
        self.feat5 = Feature("First word is 0", True)
        self.feat6 = Feature("Second word is 0", True)
        self.feat7 = Feature("Third word is 0", True)
        self.feat8 = Feature("Last word is 0", True)

        self.featSet = FeatureSet({self.feat1, self.feat2, self.feat3, self.feat4, self.feat5, self.feat6, self.feat7,
                                   self.feat8}, None)
        self.featSetCopyExample1 = FeatureSet(self.featSet.feat, "M0")
        self.featSetCopyExample2 = FeatureSet(self.featSet.feat, "ME1")

        self.testListOfSets = [
            # M0
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 3),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 1)
            }, "positive"),
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 3),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 1)
            }, "positive"),
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 3),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 1)
            }, "positive"),
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 4),
                Feature("Amount of \"1\"", 0)
            }, "positive"),

            # ME1
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 1),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 3)
            }, "negative"),
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 2),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 2)
            }, "negative"),
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 1),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 3)
            }, "negative"),
            FeatureSet({
                Feature("Amount of characters", 7),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \"(\"", 0),
                Feature("Contains \"0\"", True),
                Feature("Amount of \"0\"", 2),
                Feature("Contains \"1\"", True),
                Feature("Amount of \"1\"", 2)
            }, "negative")
        ]
        self.testClassifier = TweetClassifier.train(self.testListOfSets)

        # ALL TRAIN TESTING MATERIAL
        self.a = "a"
        self.ab = "a b b"
        self.testTrainListOfSets = [
            FeatureSet({
                Feature("Amount of characters", 1),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \")\"", 0),
                Feature("Contains \"a\"", True),
                Feature("Amount of \"a\"", 1)
            }, "positive"),
            FeatureSet({
                Feature("Amount of characters", 5),
                Feature("Amount of \"(\"", 0),
                Feature("Amount of \")\"", 0),
                Feature("Contains \"a\"", True),
                Feature("Amount of \"a\"", 1),
                Feature("Contains \"b\"", True),
                Feature("Amount of \"b\"", 2),
            }, "negative"),
        ]

        # potential problems:
        # need to assign keys as feature.name not feature?
        # does the order of the dict matter?
        testing_dict = {
            Feature("Amount of \")\"", 0): [1.0, 1.0],
            Feature("Contains \"a\"", True): [1.0, 1.0],
            Feature("Amount of characters", 1): [1.0, 0.0],
            Feature("Amount of \"a\"", 1): [1.0, 1.0],
            Feature("Amount of \"(\"", 0): [1.0, 1.0],
            Feature("Amount of characters", 5): [0.0, 1.0],
            Feature("Contains \"b\"", True): [0.0, 1.0],
            Feature("Amount of \"b\"", 2): [0.0, 1.0],
        }
        self.constructed_classifier1 = TweetClassifier({})
        self.constructed_classifier2 = TweetClassifier(testing_dict)  # the real one

    def test_train_1(self):
        trained_classifier = TweetClassifier.train(self.testTrainListOfSets)

        self.assertNotEquals(trained_classifier, self.constructed_classifier1)
        self.assertEquals(trained_classifier.get_probability_dict(),
                          self.constructed_classifier2.get_probability_dict())
        # for item in trained_classifier.get_probability_dict().keys():
        #     print("The feature name:", item, "| The feature prob:", trained_classifier.get_probability_dict()[item])

    def test_gamma_1(self):
        self.testText1 = ["0", "0", "0", "0"]
        self.setOfFeatures1 = {
            Feature("Amount of characters", 7),
            Feature("Amount of \"(\"", 0),
            Feature("Amount of \"(\"", 0),
            Feature("Contains \"0\"", True),
            Feature("Amount of \"0\"", 4),
            Feature("Contains \"1\"", False),
            Feature("Amount of \"1\"", 0)
        }
        self.FeatureSet1 = FeatureSet(self.setOfFeatures1)
        output1 = self.testClassifier.gamma(self.FeatureSet1)
        self.assertEqual(output1, "positive, gamma = " + str(1/32))

    def test_gamma_2(self):
        self.testText3 = ["0", "0", "1", "1"]
        self.setOfFeatures3 = {
            Feature("Amount of characters", 7),
            Feature("Amount of \"(\"", 0),
            Feature("Amount of \"(\"", 0),
            Feature("Contains \"0\"", True),
            Feature("Amount of \"0\"", 2),
            Feature("Contains \"1\"", True),
            Feature("Amount of \"1\"", 2)
        }
        self.FeatureSet3 = FeatureSet(self.setOfFeatures3)
        output3 = self.testClassifier.gamma(self.FeatureSet3)
        self.assertEqual(output3, "negative, gamma = " + str(1/8))  # Our model messes up here because of low sample!

    def test_gamma_3(self):
        self.testText4 = ["1", "1", "0", "0"]
        self.setOfFeatures4 = {
            Feature("Amount of characters", 7),
            Feature("Amount of \"(\"", 0),
            Feature("Amount of \"(\"", 0),
            Feature("Contains \"0\"", True),
            Feature("Amount of \"0\"", 2),
            Feature("Contains \"1\"", True),
            Feature("Amount of \"1\"", 2)
        }
        self.FeatureSet4 = FeatureSet(self.setOfFeatures4)
        output4 = self.testClassifier.gamma(self.FeatureSet4)
        self.assertEqual(output4, "negative, gamma = " + str(1/8))

    def test_gamma_4(self):
        self.testText5 = ["0", "1", "0", "1"]
        self.setOfFeatures5 = {
            Feature("Amount of characters", 7),
            Feature("Amount of \"(\"", 0),
            Feature("Amount of \"(\"", 0),
            Feature("Contains \"0\"", True),
            Feature("Amount of \"0\"", 2),
            Feature("Contains \"1\"", True),
            Feature("Amount of \"1\"", 2)
        }
        self.FeatureSet5 = FeatureSet(self.setOfFeatures5)
        output5 = self.testClassifier.gamma(self.FeatureSet5)
        self.assertEqual(output5, "negative, gamma = " + str(1/8))

    def test_present_features_0(self):
        self.testClassifier.present_features(10)
        self.constructed_classifier2.present_features(4)

    def test_present_features_1(self):
        self.testClassifier.present_features()
        self.assertEqual(self.testClassifier.order_features(), "Most informative features:\n1.  Contains \"1\" = True             negative : positive         1.33 : 1")

    def test_present_features_2(self):
        self.testClassifier.present_features()
        self.assertEqual(self.testClassifier.order_features(1),"Most informative features:\n1.  Contains \"1\" = True             negative : positive         1.33 : 1")

    def test_present_features_3(self):
        self.testClassifier.present_features()
        top10 = self.testClassifier.order_features(10)
        self.assertIn("Amount of \"1\" = 1               positive : negative            1 : 1", top10)


if __name__ == '__main__':
    unittest.main()
