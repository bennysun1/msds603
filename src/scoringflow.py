from metaflow import FlowSpec, step, Flow, Parameter, JSONType

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
        Starting point: Load the trained model.
        Gets the best model from the latest training run.
        """
        run = Flow('ModelTrainingFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
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