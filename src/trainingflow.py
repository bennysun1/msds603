from metaflow import FlowSpec, step

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

        self.model = KNeighborsClassifier()
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def train_svm(self):
        """
        Trains a Support Vector Machine classifier.
        Uses polynomial kernel which often works well for non-linear data.
        """
        from sklearn import svm

        self.model = svm.SVC(kernel='poly')
        self.model.fit(self.train_data, self.train_labels)
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
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Print results and save the best model.
        """
        print('Model Scores:')
        print('\n'.join('%s: %.3f' % res for res in self.results))
        print('\nBest Model:', self.model)


if __name__=='__main__':
    ModelTrainingFlow()