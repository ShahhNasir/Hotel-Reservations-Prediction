import numpy as np
import joblib
from config.path_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='F:/Data Science Projects/Hotel-Reservations-Prediction/template')  

# Load your trained model
loaded_model = joblib.load(MODEL_OUTPUT_PATH)  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            # Get all form data with proper field names (matching your HTML)
            lead_time = int(request.form.get("lead_time"))
            no_of_special_request = int(request.form.get("no_of_special_request"))
            avg_price_per_room = float(request.form.get("avg_price_per_room"))  # Changed to float
            arrival_month = int(request.form.get("arrival_month"))
            arrival_date = int(request.form.get("arrival_date"))
            market_segment_type = int(request.form.get("market_segment_type"))
            no_of_week_nights = int(request.form.get("no_of_week_nights"))
            no_of_weekend_nights = int(request.form.get("no_of_weekend_nights"))
            type_of_meal_plan = int(request.form.get("type_of_meal_plan"))
            room_type_reserved = int(request.form.get("room_type_reserved"))
            
            # Create feature array in the exact order your model expects
            features = np.array([
                lead_time,
                avg_price_per_room,
                no_of_special_request,
                arrival_month,
                arrival_date,
                market_segment_type,
                no_of_week_nights,
                no_of_weekend_nights,
                type_of_meal_plan,
                room_type_reserved
            ]).reshape(1, -1)  # Reshape for single prediction
            
            prediction = loaded_model.predict(features)[0]
            
        except KeyError as e:
            error = f"Missing field: {str(e)}"
        except ValueError as e:
            error = f"Invalid input: {str(e)}"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template(
        'index.html',
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)