import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder



# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)

        # Ensure required columns exist
        required_columns = ["Source", "Destination", "Route", "Total_Stops", "Airline", "Intermediate_Stops"]
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Column '{col}' not found in the dataset.")
                return None

        # Convert Total_Stops to numeric format
        data["Total_Stops"] = (
            data["Total_Stops"]
            .astype(str)
            .str.extract(r"(\d+)")  # Extract the number of stops
            .fillna(0)
            .astype(int)
        )

        # Ensure Route is of string type
        data["Route"] = data["Route"].astype(str)

        # Handle missing Intermediate_Stops column if it doesn't exist
        if "Intermediate_Stops" not in data.columns:
            data["Intermediate_Stops"] = "NIL"

        # Create the formatted route column
        data['Formatted_Route'] = data.apply(lambda row: format_route(row['Route'], row['Intermediate_Stops']), axis=1)

        return data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def format_route(route, intermediate_stops):
    # Ensure route is a string and split it by " to "
    route_parts = route.split(" to ")
    
    # First and last stop (source and destination)
    start = route_parts[0]
    end = route_parts[-1]
    
    # Check if intermediate_stops is a string or integer
    if isinstance(intermediate_stops, str):
        # If it's a string, split it by commas if there are multiple stops
        if intermediate_stops != "NIL":  # Ignore NIL as there are no intermediate stops
            intermediate_parts = intermediate_stops.split(",")  # multiple stops separated by commas
        else:
            intermediate_parts = []
    elif isinstance(intermediate_stops, int) and intermediate_stops != 0:
        # If intermediate_stops is an integer and not zero, you might want to convert it to a string
        intermediate_parts = [str(intermediate_stops)]
    else:
        intermediate_parts = []  # No intermediate stops or NIL

    # Format the route properly
    if intermediate_parts:
        formatted_route = f"{start} to {' to '.join(intermediate_parts)} to {end}"
    else:
        formatted_route = f"{start} to {end}"

    return formatted_route


def time_to_minutes(time_str):
    """Convert time to minutes since midnight."""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except ValueError:
        return 0  # Handle unexpected formats


def flight_price_regression(user_input):
    # Ensure the user input is processed correctly
    user_input_processed = user_input.copy()

    # Convert 'Dep_Time' and 'Arrival_Time' to minutes since midnight
    user_input_processed['Dep_Time'] = time_to_minutes(user_input_processed['Dep_Time'])
    user_input_processed['Arrival_Time'] = time_to_minutes(user_input_processed['Arrival_Time'])

    # Ensure that user_input_processed is a DataFrame, even if it's a dictionary
    if isinstance(user_input_processed, dict):
        user_input_df = pd.DataFrame([user_input_processed])  # Convert dict to DataFrame
    else:
        user_input_df = user_input_processed  # If it's already a DataFrame, use it directly

    # Create the 'Formatted_Route' column for user input
    user_input_df['Formatted_Route'] = user_input_df['Route']
   

    # Load the dataset for training
    df = pd.read_csv("Flight_Price_Cleaned.csv")
    
    # Ensure 'Dep_Time' and 'Arrival_Time' are converted properly in the dataset
    df['Dep_Time'] = df['Dep_Time'].apply(time_to_minutes)
    df['Arrival_Time'] = df['Arrival_Time'].apply(time_to_minutes)
    
    # Ensure 'Total_Stops' is integer
    df['Total_Stops'] = df['Total_Stops'].fillna(0).astype(int)
    
    # Create the 'Formatted_Route' column in the training data
    df['Formatted_Route'] = df.apply(lambda row: format_route(row['Route'], row['Intermediate_Stops']), axis=1)

    # Debugging: Print a sample of formatted routes in the training data
    #st.write(f"Training Data Formatted Route (Sample): {df['Formatted_Route'].head()}")

    # Feature and target selection
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Convert categorical features using OneHotEncoder
    categorical_columns = ['Airline', 'Source', 'Destination', 'Formatted_Route', 'Additional_Info']
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
    file_path = r"C:\Users\ashwi\GUVI_Projects\Flight_Project\Passenger_Satisfaction_Cleaned.csv"  
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
# Main Streamlit app
def main():
    st.title("Flight Price Prediction Application")

    # Create tabs
    tab1, tab2 = st.tabs(["Regression", "Classification"])
    with tab1:
        # Load the dataset
        file_path = r"C:\Users\ashwi\GUVI_Projects\Flight_Project\Flight_Price_Cleaned.csv"
        flight_price_data = load_data(file_path)

        if flight_price_data is not None:
            # Dropdown for Airline
            airline = st.selectbox(
                "Select Airline Type",
                flight_price_data["Airline"].unique(),
                placeholder="Select Airline Type",
            )

            # Dropdowns for Source and Destination
            source = st.selectbox(
                "Select Source",
                flight_price_data["Source"].unique(),
                placeholder="Select Source",
            )
            destination = st.selectbox(
                "Select Destination",
                flight_price_data["Destination"].unique(),
                placeholder="Select Destination",
            )

            # Dropdown for Total Stops
            total_stops = st.selectbox(
                "Select Total Stops",
                sorted(flight_price_data["Total_Stops"].unique()),
                placeholder="Select Total Stops",
            )

            # Apply filters dynamically
            filtered_data = flight_price_data[(
                flight_price_data["Airline"] == airline) 
                & (flight_price_data["Source"] == source)
                & (flight_price_data["Destination"] == destination)
                & (flight_price_data["Total_Stops"] == total_stops)
            ]
            
            # Check if any filtered data exists
            if not filtered_data.empty:
                # Format all routes in the filtered data and get unique routes
                formatted_route_options = [
                    format_route(row["Route"], row["Intermediate_Stops"])
                    for _, row in filtered_data.iterrows()
                ]

                # Get unique formatted routes
                unique_formatted_route_options = list(set(formatted_route_options))

                # Route dropdown with all unique formatted routes
                route = st.selectbox(
                    "Select Route", sorted(unique_formatted_route_options), placeholder="Select Route"
                )
            else:
                st.warning("No routes available for the selected filters.")

            additional_info = st.selectbox("Enter Additional Info", flight_price_data["Additional_Info"].unique(), placeholder="Enter other details")

            # Ask the user to input time in HH:MM format
            Dept_time_ip = st.text_input("Enter Departure Time (HH:MM):", value="00:00")
            dep_time = datetime.strptime(Dept_time_ip, "%H:%M").time()

            arr_time_ip = st.text_input("Enter Arrival Time (HH:MM):", value="00:00")
            arrival_time = datetime.strptime(arr_time_ip, "%H:%M").time()

            intermediate_stops = total_stops

            # Check if all required fields except 'route' are filled
            
            try:
                # Check if 'route' has a value
                if route:
                    # Display the "Run Regression" button only when 'route' has a value
                    if st.button("🔍 Run Regression"):
                        user_input = {
                            "Airline": airline,
                            "Source": source,
                            "Destination": destination,
                            "Route": route,
                            "Dep_Time": Dept_time_ip,
                            "Arrival_Time": arr_time_ip,
                            "Intermediate_Stops": intermediate_stops,
                            "Additional_Info": additional_info,
                            "Total_Stops": total_stops,
                        }
                        flight_price_regression(user_input)
                else:
                    # If 'route' is empty or None, show a warning
                    st.warning("Please select a valid route before proceeding.")
            except NameError:
                # Handle cases where 'route' is not defined or not assigned
                st.warning("Fields are not defined. Please make a selection.")
            except Exception as e:
                # Catch any other unexpected exceptions
                st.error(f"An unexpected error occurred: {e}")
                
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
        gender = st.selectbox("Select Gender", ["Female", "Male"], placeholder="Select Gender")
        customer_type = st.selectbox("Select Customer Type", ["Loyal Customer", "Disloyal Customer"], placeholder="Select Customer Loyalty")
        travel_type = st.selectbox("Select Type of Travel", ["Business travel", "Personal Travel"], placeholder="Select Travel Plan")
        flight_class = st.selectbox("Select Flight Class", ["Business", "Eco", "Eco Plus"], placeholder="Select Class")

        if st.button("🔍 Run Classification"):
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
                "Gender": gender,
                "Customer Type": customer_type,
                "Type of Travel": travel_type,
                "Flight Class": flight_class
            }
            customer_satisfaction_classification(user_input)

if __name__ == "__main__":
    main()


