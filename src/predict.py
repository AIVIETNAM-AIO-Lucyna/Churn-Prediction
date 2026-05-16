import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

TRAIN_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_data.csv"
)


class ChurnPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        train_df = pd.read_csv(
            TRAIN_DATA_PATH
        )

        self.feature_columns = train_df.drop(
            ["CustomerID", "Churn"],
            axis=1,
            errors="ignore"
        ).columns


    def preprocess(self, input_df):
        input_df = input_df.drop(
            ["CustomerID"],
            axis=1,
            errors="ignore"
        )

        processed = pd.get_dummies(
            input_df,
            drop_first=False
        )

        processed = processed.reindex(
            columns=self.feature_columns,
            fill_value=0
        )
        processed = processed.astype(float)

        return processed


    def predict(self, input_df):

        processed_input = self.preprocess(
            input_df
        )

        print(processed_input.T)

        prediction = self.model.predict(
            processed_input
        )[0]

        probability = self.model.predict_proba(
            processed_input
        )[0][1]

        return prediction, probability


predictor = ChurnPredictor()


def predict_customer(input_df):

    prediction, probability = predictor.predict(
        input_df
    )

    return prediction, probability