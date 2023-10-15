import sys
import os
import pickle
import unittest
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

model_pkl_path = os.path.join(parent_dir, 'Model', 'model.pkl')
with open(model_pkl_path, 'rb') as file:
    model_pkl = pickle.load(file)

class TestModel(unittest.TestCase):

    def test_prediction(self):
        test_input = [5.1, 3.5, 1.4, 0.2] 

        prediction = model_pkl.predict([test_input])

        expected_output = "Setosa" 

        self.assertEqual(prediction, expected_output, "Prediction doesn't match expected result")

if __name__ == '__main__':
    unittest.main()
