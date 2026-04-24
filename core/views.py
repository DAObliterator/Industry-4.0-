from django.shortcuts import render, redirect
from .models import MachineData
import joblib
import os

# 1. LOAD THE BRAIN (Happens once when server starts)
# This un-serializes the .pkl file back into a Python object variable called 'model'
model_path = os.path.join(os.path.abspath(os.path.dirname(__name__)), 'machine_model.pkl')
model = joblib.load(model_path)

def dashboard(request):
    # HANDLING A DATA ENTRY (POST)
    if request.method == 'POST':
        # Extract values from the <input> tags in your HTML
        # request.POST is a dict-like object containing user input
        air_temp = float(request.POST.get('air_temp'))
        proc_temp = float(request.POST.get('proc_temp'))
        speed = float(request.POST.get('speed'))
        torque = float(request.POST.get('torque'))
        wear = float(request.POST.get('wear'))

        # PREDICTION: We wrap the 5 values in a 2D list because the model expects a "table"
        # Input: [[300, 310, 1500, 40, 5]] -> Output: [0] or [1]
        prediction_result = model.predict([[air_temp, proc_temp, speed, torque, wear]])
        
        # Access the first element of the result array (the 0 or 1)
        final_answer = int(prediction_result[0])

        # DB WRITE: Save the inputs + the brain's answer to Supabase
        MachineData.objects.create(
            air_temperature=air_temp,
            process_temperature=proc_temp,
            rotational_speed=speed,
            torque=torque,
            tool_wear=wear,
            prediction=final_answer
        )
        return redirect('dashboard') # Refresh page to show new data

    # HANDLING THE PAGE VIEW (GET)
    # Fetch last 50 rows from Supabase to show on the chart/table
    readings = MachineData.objects.all().order_by('-timestamp')[:50]
    
    return render(request, 'dashboard.html', {'readings': readings})
