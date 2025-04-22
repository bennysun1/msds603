from metaflow import FlowSpec, step, Flow, Parameter, JSONType
import mlflow

class ModelScoringFlow(FlowSpec):
    """Flow for making predictions using the trained model."""
    
    # Input parameter for the data point to predict
    vector = Parameter(
        'vector',
        type=JSONType,
        required=True,
        help='Input vector for prediction. Must match the format of training data.'
    )

    @step
    def start(self):
        """
        Starting point: Load the trained model from MLFlow.
        Gets the latest version of the registered model.
        """
        # Set up MLFlow tracking
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        
        # Load the model from MLFlow Model Registry
        model_name = "wine-classifier"
        model_version = 1  # You might want to make this a parameter
        
        self.model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        
        print("Input vector:", self.vector)
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Make and output predictions.
        Uses the loaded model to predict the class of the input vector.
        """
        print('Using Model:', self.model)
        print('Predicted class:', self.model.predict([self.vector])[0])


if __name__=='__main__':
    ModelScoringFlow() 