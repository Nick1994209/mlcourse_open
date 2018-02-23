from sklearn.model_selection import train_test_split


print(train_test_split([10, 11, 12,1,1,1,1,21,12,31,1,213,123,], test_size=0.3, random_state=17))
X_train, X_valid, y_train, y_valid = train_test_split([10, 11, 12,1,1,1,1,21,12,31,1,213,123,], test_size=0.3, random_state=17)