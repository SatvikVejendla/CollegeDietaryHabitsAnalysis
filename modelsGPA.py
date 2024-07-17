import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("processed_data.csv")

print(df.head())


X = df.drop(columns=["GPA"])

y = df["GPA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
train_pred = model.predict(X_train)

from sklearn.metrics import mean_squared_error

print("Mean Squared Error: " + str(mean_squared_error(y_test, y_pred)))
print("Mean Squared Error (train): " + str(mean_squared_error(y_train, train_pred)))


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

train_pred = model.predict(X_train)

print("Mean Squared Error: " + str(mean_squared_error(y_test, y_pred)))

print("Mean Squared Error (train): " + str(mean_squared_error(y_train, train_pred)))


from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_pred = model.predict(X_train)

print("Mean Squared Error: " + str(mean_squared_error(y_test, y_pred)))

print("Mean Squared Error (train): " + str(mean_squared_error(y_train, train_pred)))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold


model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dense(64, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),verbose=2)

y_pred = model.predict(X_test)

train_pred = model.predict(X_train)

print("Mean Squared Error: " + str(mean_squared_error(y_test, y_pred)))

print("Mean Squared Error (train): " + str(mean_squared_error(y_train, train_pred)))

model.save("modelGPA.keras")