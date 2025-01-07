import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def time_to_minutes(time_str):
    """Convert time to minutes since midnight."""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except ValueError:
        return 0  # Handle unexpected formats
    
def flight_price_regression(user_input):
    user_input_processed = user_input.copy()

    # Convert 'Dep_Time' and 'Arrival_Time' to minutes since midnight
    user_input_processed['Dep_Time'] = time_to_minutes(user_input_processed['Dep_Time'])
    user_input_processed['Arrival_Time'] = time_to_minutes(user_input_processed['Arrival_Time'])

    # Ensure that user_input_processed is a DataFrame, even if it's a dictionary
    if isinstance(user_input_processed, dict):
        user_input_df = pd.DataFrame([user_input_processed])  # Convert dict to DataFrame
    else:
        user_input_df = user_input_processed  # If it's already a DataFrame, use it directly

    # Check if 'Total_Stops' is in the DataFrame columns
    if 'Total_Stops' in user_input_df.columns:
        user_input_df['Total_Stops'] = int(user_input_df['Total_Stops'])
    else:
        st.warning("'Total_Stops' column not found in the user input. Assigning default value.")
        user_input_df['Total_Stops'] = 0  # Or any default value

    # Load the dataset for training
    df = pd.read_csv("Flight_Price_Cleaned.csv")
    
    # Ensure 'Dep_Time' and 'Arrival_Time' are converted properly in the dataset
    df['Dep_Time'] = df['Dep_Time'].apply(time_to_minutes)
    df['Arrival_Time'] = df['Arrival_Time'].apply(time_to_minutes)
    
    # Ensure 'Total_Stops' is integer
    df['Total_Stops'] = df['Total_Stops'].fillna(0).astype(int)
    
    # Feature and target selection
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Convert categorical features using OneHotEncoder
    categorical_columns = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']
    numerical_columns = ['Dep_Time', 'Arrival_Time', 'Total_Stops']
    
    # Define the transformer for categorical and numerical features
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
            ('num', StandardScaler(), numerical_columns)
        ]
    )
    
    # Create a pipeline with the transformer and the RandomForest model
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X, y)
    
    # Predict the flight price for the user input
    predicted_price = pipeline.predict(user_input_df)
    
    st.write("Predicted flight price:", f"{predicted_price[0]:.2f}")


def customer_satisfaction_classification(user_input):
    file_path = r"C:\Users\ashwi\GUVI_Projects\Flight Project\Passenger_Satisfaction_Cleaned.csv"  # Update with your file path
    df = pd.read_csv(file_path)

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Fill missing values for numeric columns with the mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill missing values for categorical columns with 'Unknown'
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Encoding categorical variables using Label Encoding
    label_cols = ['Gender_Female', 'Gender_Male', 'Customer Type_Loyal Customer', 'Customer Type_disloyal Customer',
                'Type of Travel_Business travel', 'Type of Travel_Personal Travel', 'Class_Business', 'Class_Eco', 'Class_Eco Plus']

    # Apply Label Encoding to categorical columns
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # Encode the target column 'satisfaction'
    df['satisfaction'] = le.fit_transform(df['satisfaction'])

    # Feature columns and target column
    X = df.drop(columns=['satisfaction'])  # Features
    y = df['satisfaction']  # Target

    # Standardize the numerical features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Initialize Random Forest model
    random_forest = RandomForestClassifier(random_state=42)

    # Train the Random Forest model
    random_forest.fit(X, y)

    # Ensure the input features match the model's features (order should match)
    user_input_df = pd.DataFrame([user_input])
    
    # Align the user input with the model's input features (matching columns)
    # Apply transformations to the user input based on the trained model
    user_input_df[numeric_cols] = scaler.transform(user_input_df[numeric_cols])

    # Ensure the user input has the same columns as the model (including one-hot encoded categorical columns)
    missing_cols = set(X.columns) - set(user_input_df.columns)
    for col in missing_cols:
        user_input_df[col] = 0  # Add missing columns with default value (e.g., 0 for encoded categories)

    # Reorder the columns to match the model
    user_input_df = user_input_df[X.columns]

    # Predict the satisfaction for the user input
    user_prediction = random_forest.predict(user_input_df)

    # Decode the prediction (convert numerical labels back to original labels)
    predicted_satisfaction = le.inverse_transform(user_prediction)

    # Output the predicted result
    st.write("Predicted satisfaction:", f"{predicted_satisfaction[0]}")

def main():
    st.title("Flight Price Prediction Application")

    # Create tabs
    tab1, tab2 = st.tabs(["Regression", "Classification"])

    with tab1:
        st.header("Regression Tab")
        
        # User inputs for regression
        airline = st.selectbox("Select Airline", ["Air Asia", "Air India", "GoAir", "IndiGo", "Jet Airways", "Jet Airways Business", "Multiple Carriers", "Multiple Carriers Premium Economy", "Spicejet", "Trujet", "Vistara", "Vistara Premium economy"])
        source = st.selectbox("Select Source", ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"])
        destination = st.selectbox("Select Destination", ["Bangalore", "Cochin", "Hyderabad", "New Delhi", "Delhi", "Kolkata"])
        total_stops = st.selectbox("Select Total Stops", ["0", "1", "2", "3"])

        # Additional inputs for regression
        route = st.text_input("Enter Route (e.g., DEL to BOM)")
        additional_info = st.text_input("Enter Additional Info")

        # Ask the user to input time in HH:MM format
        Dept_time_ip = st.text_input("Enter Departure Time(HH:MM):", value="00:00")
        dep_time = datetime.strptime(Dept_time_ip, "%H:%M").time()

        arr_time_ip = st.text_input("Enter Arrival Time(HH:MM):", value="00:00")
        arrival_time = datetime.strptime(arr_time_ip, "%H:%M").time()

        intermediate_stops = st.number_input("Enter intermediate stops", min_value=0)
        if st.button("🔍 Run Regression"):
            user_input = {
                "Airline": airline,
                "Source": source,
                "Destination": destination,
                "Route": route,
                "Dep_Time": Dept_time_ip,
                "Arrival_Time": arr_time_ip,  # Corrected to match the expected key
                "Intermediate Stops": intermediate_stops,
                "Additional_Info": additional_info,
                "Total_Stops": total_stops
            }
            flight_price_regression(user_input)

    with tab2:
        st.header("Classification Tab")
        
        # Inputs for classification
        age = st.number_input("Enter Age", min_value=0)
        flight_distance = st.number_input("Enter Flight Distance", min_value=0)

        # Ratings inputs
        st.subheader("Customer Ratings")
        inflight_wifi_service = st.slider("Inflight wifi service", 0, 5)
        departure_arrival_convenient = st.slider("Departure/Arrival time convenient", 0, 5)
        ease_of_online_booking = st.slider("Ease of Online booking", 0, 5)
        gate_location = st.slider("Gate location", 0, 5)
        food_and_drink = st.slider("Food and drink", 0, 5)
        online_boarding = st.slider("Online boarding", 0, 5)
        seat_comfort = st.slider("Seat comfort", 0, 5)
        inflight_entertainment = st.slider("Inflight entertainment", 0, 5)
        onboard_service = st.slider("On-board service", 0, 5)
        leg_room_service = st.slider("Leg room service", 0, 5)
        baggage_handling = st.slider("Baggage handling", 0, 5)
        checkin_service = st.slider("Checkin service", 0, 5)
        inflight_service = st.slider("Inflight service", 0, 5)
        cleanliness = st.slider("Cleanliness", 0, 5)
        departure_delay = st.number_input("Departure Delay in Minutes", min_value=0)
        arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0)

        # Dropdowns for categorical inputs
        gender = st.selectbox("Select Gender", ["Female", "Male"])
        gender_female = 1 if gender == "Female" else 0
        gender_male = 1 if gender == "Male" else 0

        customer_type = st.selectbox("Select Customer Type", ["Loyal Customer", "Disloyal Customer"])
        customer_type_loyal = 1 if customer_type == "Loyal Customer" else 0
        customer_type_disloyal = 1 if customer_type == "Disloyal Customer" else 0

        travel_type = st.selectbox("Select Type of Travel", ["Business travel", "Personal Travel"])
        travel_business = 1 if travel_type == "Business travel" else 0
        travel_personal = 1 if travel_type == "Personal Travel" else 0

        flight_class = st.selectbox("Select Flight Class", ["Business", "Eco", "Eco Plus"])
        flight_class_business = 1 if flight_class == "Business" else 0
        flight_class_eco = 1 if flight_class == "Eco" else 0
        flight_class_eco_plus = 1 if flight_class == "Eco Plus" else 0

        if st.button("🔍 Run Classification"):# Construct the input dictionary
            user_input = {
                "Age": age,
                "Flight Distance": flight_distance,
                "Inflight wifi service": inflight_wifi_service,
                "Departure/Arrival time convenient": departure_arrival_convenient,
                "Ease of Online booking": ease_of_online_booking,
                "Gate location": gate_location,
                "Food and drink": food_and_drink,
                "Online boarding": online_boarding,
                "Seat comfort": seat_comfort,
                "Inflight entertainment": inflight_entertainment,
                "On-board service": onboard_service,
                "Leg room service": leg_room_service,
                "Baggage handling": baggage_handling,
                "Checkin service": checkin_service,
                "Inflight service": inflight_service,
                "Cleanliness": cleanliness,
                "Departure Delay in Minutes": departure_delay,
                "Arrival Delay in Minutes": arrival_delay,
                "Gender_Female": gender_female,
                "Gender_Male": gender_male,
                "Customer Type_Loyal Customer": customer_type_loyal,
                "Customer Type_disloyal Customer": customer_type_disloyal,
                "Type of Travel_Business travel": travel_business,
                "Type of Travel_Personal Travel": travel_personal,
                "Class_Business": flight_class_business,
                "Class_Eco": flight_class_eco,
                "Class_Eco Plus": flight_class_eco_plus
            }
            customer_satisfaction_classification(user_input)

if __name__ == "__main__":
    main()
