
import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('employee_burnout_model.pkl')

def predict_burnout(company_type, wfh_setup, resource_allocation, mental_fatigue_score):
    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        "Company Type": company_type,
        "WFH Setup Available": wfh_setup,
        "Resource Allocation": resource_allocation,
        "Mental Fatigue Score": mental_fatigue_score
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return f"Predicted Burnout Rate: {prediction:.2f}"

# Define Gradio interface
inputs = [
    gr.Textbox(label="Company Type", placeholder="Enter Company Type (e.g., Service/Product)"),
    gr.Textbox(label="WFH Setup Available", placeholder="Yes/No"),
    gr.Slider(label="Resource Allocation", minimum=0, maximum=10, step=1),
    gr.Slider(label="Mental Fatigue Score", minimum=0.0, maximum=10.0, step=0.1),
]
output = gr.Textbox(label="Burnout Prediction")

app = gr.Interface(fn=predict_burnout, inputs=inputs, outputs=output, title="Employee Burnout Predictor")
app.launch()
