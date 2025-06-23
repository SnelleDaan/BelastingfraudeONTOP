import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


#Opdracht 2a)
def LoadAndVisualizeData():
    # 1. Data inlezen
    data = pd.read_csv("Eindopdracht/shaft_radius.csv")

    # 2. Train/test split (bijv. 80/20)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    # 3. Plot van trainingsdata
    sorted_train_data = train_data.sort_values(by="measurement_index").copy()
    plt.plot(sorted_train_data["measurement_index"], sorted_train_data["shaft_radius"], label="Train data", color="blue")
    plt.title("Shaft Radius over Time (Train Set)")
    plt.xlabel("Time [hours]")
    plt.ylabel("Shaft Radius [m]")
    plt.legend()
    plt.show()
    return train_data, test_data

train_data, test_data = LoadAndVisualizeData()

#Opdracht 2b)
def Opdracht2(train_data, test_data):
    # Feature en target definiëren
    X_train = train_data[["measurement_index"]].values
    y_train = train_data["shaft_radius"].values
    X_test = test_data[["measurement_index"]].values
    y_test = test_data["shaft_radius"].values


    best_depth = None
    best_r2 = -float('inf')
    for depth in range(1, 21):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Depth: {depth:2d} | Test R²: {r2:.4f} | MSE: {mse:.20f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_depth = depth

    print(f"\nBeste diepte: {best_depth} met R² = {best_r2:.4f}")



    model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sorted_test_data = test_data.sort_values(by="measurement_index").copy()
    plt.scatter(sorted_test_data["measurement_index"], sorted_test_data["shaft_radius"], label='Actual', color='blue', alpha=0.5)
    plt.scatter(X_test.flatten(), y_pred, label='Prediction', color='red', alpha=0.5)
    plt.xlabel('Measurement Index')
    plt.ylabel('Shaft Radius')
    plt.title('Decision Tree vs Actual Radius')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


Opdracht2(train_data, test_data)
