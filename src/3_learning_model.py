#%%
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
import subprocess
import configuration
from sklearn.impute import SimpleImputer
import numpy as np
import os
from sklearn.metrics import accuracy_score

# Generate feature and output vectors from act states.
df_act_states = pd.read_csv(configuration.states_csv)
# quick remove of created field
output_list = ["entity_id", "state"]
feature_list = list(df_act_states.columns)
# Remove output vectors
for output in output_list:
    feature_list.remove(output)

for actuators in configuration.actuators:
    df_act = df_act_states[df_act_states["entity_id"] == actuators]
    output_vector = df_act["entity_id"] + df_act["state"]
    feature_vector = df_act[feature_list]

    # Split into random training and test set
    X = feature_vector
    y = output_vector

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=170
    )  # 2)

    model_tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    model_logs = LogisticRegression(random_state=99, max_iter=1000)

    model_tree.fit(X, y)
    model_logs.fit(X, y)

    def visualize_tree(tree, feature_names):
        """Create tree png using graphviz.

        Args
        ----
        tree -- scikit-learn DecsisionTree.
        feature_names -- list of feature names.
        """
        with open(f"{actuators}.dot", "w") as f:
            export_graphviz(tree, out_file=f, feature_names=feature_names)

        command = ["dot", "-Tpng", f"{actuators}.dot", "-o", f"{actuators}.png"]
        try:
            subprocess.check_call(command)
            os.remove(f"{actuators}.dot")
        except:
            exit("Could not run dot, ie graphviz, to " "produce visualization")

    visualize_tree(model_tree, feature_list)

    # Get predictions of model
    y_tree_predictions = model_tree.predict(X_test)
    y_logs_predictions = model_logs.predict(X_test)
    # Extract predictions for each output variable and calculate accuracy and f1 score
    print(accuracy_score(y_test, y_tree_predictions) * 100)
    print(accuracy_score(y_test, y_logs_predictions) * 100)
