import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, path: str, target: str = "estimated_stock_pct", split: float = 0.75, k: int = 10):
        self.path = path
        self.target = target
        self.split = split
        self.k = k
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """
        This function takes a path string to a CSV file and loads it into
        a Pandas DataFrame.

        :return     df: pd.DataFrame
        """
        df = pd.read_csv(self.path)
        df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
        return df

    def create_target_and_predictors(self, data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        This function takes in a Pandas DataFrame and splits the columns
        into a target column and a set of predictor variables, i.e. X & y.
        These two splits of the data will be used to train a supervised 
        machine learning model.

        :param      data: pd.DataFrame, dataframe containing data for the 
                        model

        :return     X: pd.DataFrame
                    y: pd.Series
        """
        if self.target not in data.columns:
            raise ValueError(f"Target: {self.target} is not present in the data")
        
        X = data.drop(columns=[self.target])
        y = data[self.target]
        return X, y

    def build_model(self, input_shape):
        """
        This function builds the neural network of three layers with relu activation functions

        :param      input_shape: int, number of features

        :return     model: tf.keras.Sequential
        """
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)), 
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def train_and_evaluate_model(self, X: pd.DataFrame, y: pd.Series):
        """
        This function takes the predictor and target variables and
        trains a Neural Netowrk model across K folds. Using
        cross-validation, performance metrics will be output for each
        fold during training.

        :param      X: pd.DataFrame, predictor variables
        :param      y: pd.Series, target variable

        :return     model: tf.keras.Sequential
        """
        accuracy = []

        for fold in range(self.k):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.split, random_state=42)

            # Scale the data
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # Build and train the model
            model = self.build_model(X_train.shape[1])
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            # Generate predictions
            y_pred = model.predict(X_test)

            # Compute Mean Absolute Error
            mae = mean_absolute_error(y_test, y_pred)
            accuracy.append(mae)
            print(f"Fold {fold + 1}: MAE = {mae:.3f}")

        print(f"Average MAE: {sum(accuracy) / len(accuracy):.2f}")
        return model

    def run(self):
        """
        This function executes the training pipeline of loading the prepared
        dataset from a CSV file and training the machine learning model

        :param

        :return trained_model: tf.keras.Sequential
        """
        df = self.load_data()
        X, y = self.create_target_and_predictors(data=df)
        trained_model = self.train_and_evaluate_model(X=X, y=y)
        return trained_model

if __name__ == '__main__':
    trainer = ModelTrainer('include/dataset/cleaned_sales.csv')
    trainer.run()
