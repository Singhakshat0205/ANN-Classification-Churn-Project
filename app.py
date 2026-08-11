import tensorflow as tf
import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model("model.h5")


# ============================================================
# LOAD ENCODERS AND SCALER
# ============================================================

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Customer Churn Prediction")

st.write(
    "Enter the customer's information to predict "
    "the probability of churn."
)


# ============================================================
# USER INPUTS
# ============================================================

geography = st.selectbox(
    "Geography",
    list(onehot_encoder_geo.categories_[0])
)

gender = st.selectbox(
    "Gender",
    list(label_encoder_gender.classes_)
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=40
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})


# ============================================================
# ENCODE GENDER
# ============================================================

input_data["Gender"] = label_encoder_gender.transform(
    input_data["Gender"]
)


# ============================================================
# ONE-HOT ENCODE GEOGRAPHY
# ============================================================

geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ["Geography"]
    )
)


# ============================================================
# REMOVE ORIGINAL GEOGRAPHY
# ADD ONE-HOT ENCODED GEOGRAPHY
# ============================================================

input_data = input_data.drop(
    "Geography",
    axis=1
)

input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geo_encoded_df.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# MATCH EXACT TRAINING FEATURE ORDER
# ============================================================

input_data = input_data[
    scaler.feature_names_in_
]


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Churn"):

    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    input_data_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )


    # Convert NumPy value to normal Python float
    prediction_proba = float(
        prediction[0][0]
    )


    # --------------------------------------------------------
    # DISPLAY PROBABILITY
    # --------------------------------------------------------

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{prediction_proba * 100:.2f}%"
    )


    # --------------------------------------------------------
    # DISPLAY FINAL RESULT
    # --------------------------------------------------------

    if prediction_proba >= 0.5:

        st.error(
            "⚠️ The customer is likely to churn."
        )

    else:

        st.success(
            "✅ The customer is not likely to churn."
        )