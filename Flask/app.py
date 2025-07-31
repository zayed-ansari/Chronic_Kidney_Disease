from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Loading the model
model = pickle.load(open('CKD.pkl', 'rb'))

scaler = StandardScaler()

training_data = np.array([
    [25, 90, 0, 0, 0, 4.8, 0, 0],    # healthy
    [35, 110, 0, 0, 0, 5.2, 0, 0],     
    [45, 130, 0, 0, 0, 4.5, 1, 0],   
    [60, 150, 0, 1, 0, 4.0, 1, 0],  
    [80, 140, 1, 0, 0, 3.8, 0, 1],   # CKD
    [120, 180, 1, 0, 1, 3.2, 1, 0],
    [180, 250, 1, 1, 1, 2.5, 1, 1], 
    [200, 300, 1, 1, 1, 2.0, 1, 1], 
])

scaler.fit(training_data)
print("StandardScaler fitted and ready")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting form values
        blood_urea = float(request.form['blood_urea'])
        blood_glucose = float(request.form['blood_glucose_random'])
        aanemia = int(request.form['aanemia'])
        coronary_artery_disease = int(request.form['coronary_artery_disease'])
        pus_cell = int(request.form['pus_cell'])
        red_blood_cell_count = float(request.form['red_blood_cell_count'])  
        diabetes_mellitus = int(request.form['diabetes_mellitus'])
        pedal_edema = int(request.form['pedal_edema'])

        features = np.array([[
            blood_urea,
            blood_glucose,
            aanemia,
            coronary_artery_disease,
            pus_cell,
            red_blood_cell_count,
            diabetes_mellitus,
            pedal_edema
        ]])
        # feature = [30, 100, 0, 0 ,0 1]
        

        print(f"Raw features: {features}")
        
        features_scaled = scaler.transform(features)
        print(f"Scaled features: {features_scaled}")
        
        prediction = model.predict(features_scaled)
        output = prediction[0]
        

        print(f"Model prediction: {output}")
        
        if output == 1:
            result = "Positive (CKD detected)"
        else:
            result = "Negative (No CKD detected)"
        
        return render_template('index.html', 
                             prediction_text=f'Prediction: {result}',
                             form_data={
                                 'blood_urea': blood_urea,
                                 'blood_glucose_random': blood_glucose,
                                 'aanemia': aanemia,
                                 'coronary_artery_disease': coronary_artery_disease,
                                 'pus_cell': pus_cell,
                                 'red_blood_cell_count': red_blood_cell_count,
                                 'diabetes_mellitus': diabetes_mellitus,
                                 'pedal_edema': pedal_edema
                             })
    except Exception as e:
        print(f"Error details: {str(e)}")
        return f"Prediction Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)