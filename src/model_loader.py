import pickle
import numpy as np
# from predict_request import PredictRequest


class ModelLoader:
    def __init__(self, algorithm):
        self.death_event = {
            0: 'survival',
            1: 'dead'
        }

        if algorithm == 'logistic regression':
            self.model = pickle.load(open("logistic_regression.pkl", "rb"))

        if algorithm == 'decision tree':
            self.model = pickle.load(open("decision_tree.pkl", "rb"))

        if algorithm == 'random forest':
            self.model = pickle.load(open("random_forest.pkl", "rb"))

        if algorithm == 'extra trees':
            self.model = pickle.load(open("extra_trees.pkl", "rb"))

        if algorithm == 'neural networks':
            self.model = pickle.load(open("neural_networks.pkl", "rb"))

        if algorithm == 'ensemble voting':
            self.model = pickle.load(open("voting_classifier.pkl", "rb"))

    # def predict(self, data: PredictRequest):
    #     return self.model.predict(data)

    def predict(self, features: dict):
        X = [features['age'],
             features['anaemia'],
             features['creatinine_phosphokinase'],
             features['diabetes'],
             features['ejection_fraction'],
             features['high_blood_pressure'],
             features['platelets'],
             features['serum_creatinine'],
             features['serum_sodium'],
             features['sex'],
             features['smoking'],
             features['time']
             ]
        prediction = self.model.predict([X])
        probability = self.model.predict_proba([X]) # note that predict_proba() only work for a couple of the classifiers
        pred_class = self.death_event[prediction[0]]
        return {'prediction_class': str(pred_class),
                'probability': round(max(probability[0]), 2)
                }
