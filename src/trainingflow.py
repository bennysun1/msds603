from metaflow import FlowSpec, step
import mlflow

class ModelTrainingFlow(FlowSpec):
    """Flow for training and selecting the best ML model."""

    @step
    def start(self):
        """
        Starting point: Load and preprocess data.
        - Loads the wine dataset (TODO: replace with our own dataset)
        - Splits into train/test sets
        """
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        # Set up MLFlow tracking
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment('metaflow-wine-experiment')

        X, y = datasets.load_wine(return_X_y=True)  # TODO: Replace with our dataset
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data loaded successfully")
        self.next(self.train_knn, self.train_svm)

    @step
    def train_knn(self):
        """
        Trains a K-Nearest Neighbors classifier.
        Uses default hyperparameters for now.
        """
        from sklearn.neighbors import KNeighborsClassifier
        import mlflow.sklearn

        with mlflow.start_run(run_name='knn_training') as run:
            self.model = KNeighborsClassifier()
            self.model.fit(self.train_data, self.train_labels)
            
            # Log parameters and metrics
            mlflow.log_params({
                'model_type': 'KNN',
                'n_neighbors': self.model.n_neighbors
            })
            score = self.model.score(self.test_data, self.test_labels)
            mlflow.log_metric('test_accuracy', score)
            
            # Log the model
            mlflow.sklearn.log_model(self.model, 'knn_model')
            self.run_id = run.info.run_id
            
        self.next(self.choose_model)

    @step
    def train_svm(self):
        """
        Trains a Support Vector Machine classifier.
        Uses polynomial kernel which often works well for non-linear data.
        """
        from sklearn import svm
        import mlflow.sklearn

        with mlflow.start_run(run_name='svm_training') as run:
            self.model = svm.SVC(kernel='poly')
            self.model.fit(self.train_data, self.train_labels)
            
            # Log parameters and metrics
            mlflow.log_params({
                'model_type': 'SVM',
                'kernel': 'poly'
            })
            score = self.model.score(self.test_data, self.test_labels)
            mlflow.log_metric('test_accuracy', score)
            
            # Log the model
            mlflow.sklearn.log_model(self.model, 'svm_model')
            self.run_id = run.info.run_id
            
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        """
        Selects the best performing model based on test set accuracy.
        Receives models from both training branches and compares them.
        """
        def score(inp):
            """Helper function to compute model score."""
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        # Sort models by their score in descending order
        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]  # Select the best model
        
        # Register the best model in MLFlow
        import mlflow.sklearn
        with mlflow.start_run(run_name='model_selection'):
            model_info = mlflow.sklearn.log_model(
                self.model,
                'best_model',
                registered_model_name='wine-classifier'
            )
            self.model_uri = model_info.model_uri
            
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Print results and save the best model.
        """
        print('Model Scores:')
        print('\n'.join('%s: %.3f' % res for res in self.results))
        print('\nBest Model:', self.model)
        print('Model URI:', self.model_uri)


if __name__=='__main__':
    ModelTrainingFlow()