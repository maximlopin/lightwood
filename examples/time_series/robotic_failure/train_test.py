from sklearn.metrics import balanced_accuracy_score
import lightwood
import pandas as pd


def train_model():
    # Load some training data (default on credit, for predicting whether or not someone will default on their credit)
    df = pd.read_csv('https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/others/robotic_failure/dataset/train.csv')

    # A configuration describing the contents of the dataframe, what are the targets we want to predict and what are the features we want to use
    config = {'input_features': [
        {'name': 'id', 'type': 'numeric'}, {'name': 'time', 'type': 'numeric'},
        {'name': 'F_x', 'type': 'numeric'}, {'name': 'F_y', 'type': 'numeric'},
        {'name': 'F_z', 'type': 'numeric'}, {'name': 'T_x', 'type': 'numeric'},
        {'name': 'T_y', 'type': 'numeric'}, {'name': 'T_z', 'type': 'numeric'}],
        'output_features': [{'name': 'target', 'type': 'categorical'}],
        'mixer': {'class': lightwood.BUILTIN_MIXERS.NnMixer}}

    # Callback to log various training stats (currently the only hook into the training process)
    def train_callback(epoch, error, test_error, test_error_gradient, test_accuracy):
        print(f'We reached epoch {epoch} with error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}')

    # The actual training process
    predictor = lightwood.Predictor(config)
    # Note: If `stop_training_after_seconds` is not set, training will stop automatically once we determine the model is overfitting (we separate a testing and a training dataset internally from the dataframe given and only train on the training one, using the testing one to determine overfitting, pick the best model and evaluate model accuracy)
    predictor.learn(from_data=df, callback_on_iter=train_callback, eval_every_x_epochs=5, stop_training_after_seconds=100)

    # Save the lightwood model
    predictor.save('robotic_failure_lightwood_model.dill')


def test_model():
    # Load some testing data and extract the real values for the target column
    test = pd.read_csv('https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/others/robotic_failure/dataset/test.csv')
    real = [str(x) for x in test['target']]

    test = test.drop(columns=['target'])

    # Load the lightwood model from where we previously saved it and predict using it
    predictor = lightwood.Predictor(load_from_path='robotic_failure_lightwood_model.dill')
    predictions = predictor.predict(when_data=test)
    predicted = [str(x) for x in predictions['target']['predictions']]

    # Get the balanced accuracy score to see how well we did (in this case > 50% means better than random)
    balanced_accuracy_pct = balanced_accuracy_score(real, predicted) * 100
    print(f'Balacned accuracy score of {round(balanced_accuracy_pct,1)}%')


# Run as main
if __name__ == '__main__':
    train_model()
    test_model()
