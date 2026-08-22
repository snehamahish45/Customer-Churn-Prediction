from preprocessing import load_data, clean_data, prepare_data

df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = clean_data(df)

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
print("Preprocessing Successful!")