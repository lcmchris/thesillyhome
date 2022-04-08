import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
import configuration
import numpy as np
import os
from sklearn.metrics import accuracy_score
import pickle


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open(f"{cur_dir}/model/{actuator}.dot", "w") as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)

    command = [
        "dot",
        "-Tpng",
        f"model/{actuator}.dot",
        "-o",
        f"{cur_dir}/model/{actuator}.png",
    ]
    try:
        subprocess.check_call(command)
        os.remove(f"{cur_dir}/model/{actuator}.dot")
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    model_name = "base"
    # Generate feature and output vectors from act states.
    df_act_states = pd.read_csv(configuration.states_csv)
    # quick remove of created field
    output_list = ["entity_id", "state"]
    feature_list = list(df_act_states.columns)

    # Remove output vectors
    for output in output_list:
        feature_list.remove(output)

    for actuator in configuration.actuators:
        df_act = df_act_states[df_act_states["entity_id"] == actuator]
        output_vector = df_act["entity_id"] + "::" + df_act["state"]
        feature_vector = df_act[feature_list]
        print(len(feature_vector.columns))

        # Split into random training and test set
        X = feature_vector
        y = output_vector

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=170
        )  # 2)

        sample_weight = np.ones(len(X_train))
        sample_weight[: int(len(sample_weight) * 0.5)] = 3

        model_tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
        model_tree.fit(X_train, y_train, sample_weight)

        visualize_tree(model_tree, feature_list)

        # Get predictions of model
        y_tree_predictions = model_tree.predict(X_test)
        # Extract predictions for each output variable and calculate accuracy and f1 score
        print(
            f"{actuator} accuracy score: {accuracy_score(y_test, y_tree_predictions) * 100}"
        )

        # Save model to disk
        filename = open(
            f"{configuration.home}/src/appdaemon/apps/model/{actuator}.pickle", "wb"
        )
        pickle.dump(model_tree, filename)
    print("Completed!")
