import sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    categorical_features=[0, 1, 2], sparse=False
)  # One code feature 0,1 and 2


for device in output_devices:
    print("Training model for", device["name"], "with type", device["type"])
    X = Xs[device["device_id"]]
    y = ys[device["device_id"]]

    # Encode time values using encoder
    X = encoder.fit_transform(X)

    # Split into random training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=gmtime().tm_sec
    )  # 2)

    # Fit to model
    if device["type"] == "BinaryPowerSwitchDevice":
        model = ClassificationModel(
            n_estimators=100,
            max_features="auto",
            n_jobs=-1,
            class_weight="balanced_subsample",
        )  # {0:1,1:2}
    else:
        model = RegressionModel(n_estimators=10, max_features="log2", n_jobs=-1)

    print(
        "Cross Validation Score: ", round(mean(cross_val_score(model, X, y)) * 100, 2)
    )

    model.fit(X, y)

    y_predictions = model.predict(X_test)

    # Score predictions - calculate accuracy and f1 score
    if device["type"] == "BinaryPowerSwitchDevice":
        print(
            "Accuracy Score: {} %".format(
                round(accuracy_score(y_test, y_predictions, True) * 100, 2)
            )
        )
    else:
        print(
            "Mean Sq. Error Score: {}".format(
                round(mean_squared_error(y_test, y_predictions), 2)
            )
        )

    # Store the preprocessor and model
    joblib.dump(
        model, "models/random_forest_model_device_{}.pkl".format(device["device_id"])
    )

# Store encoder
joblib.dump(encoder, "models/feature_vector_encoder.pkl")
