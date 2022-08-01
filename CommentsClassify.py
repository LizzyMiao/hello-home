from textblob.classifiers import NaiveBayesClassifier


class CommentsClassify:

    def __init__(self, json_config_path):
        with open(json_config_path, 'r') as fp:
            self.cl = NaiveBayesClassifier(fp, format="json")

    def update_classify(self, new_data_json):
        self.cl.update(new_data_json)

    def classify_comments(self, comments):
        return self.cl.classify(comments)

    def prob_classify(self, comment, positive_label_value, negative_label_value):
        prob_dist = self.cl.prob_classify(comment)
        prob_dist_dict = {positive_label_value: prob_dist.prob(positive_label_value),
                          negative_label_value: prob_dist.prob(negative_label_value)}

        return prob_dist_dict
