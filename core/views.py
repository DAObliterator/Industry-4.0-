from django.shortcuts import render, redirect
from .models import MachineData
import joblib
import os
import numpy as np

# 1. LOAD THE TRAINED MODEL & FEATURE LIST
model_path = os.path.join(os.path.dirname(__file__), '..', 'machine_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), '..', 'model_features.pkl')

try:
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
except FileNotFoundError:
    model = None
    feature_names = None
    print("⚠️  Warning: Model files not found. Run train_model.py first!")

def compute_lagged_features(air_temp, proc_temp, speed, torque, wear):
    """
    Compute time-series lagged features based on the 5 most recent readings.
    This is the same logic used during training.
    """
    # Get last 4 readings from database
    recent = list(MachineData.objects.all().order_by('-timestamp')[:4])
    
    # Build a list of readings in chronological order (oldest to newest)
    # Convert MachineData objects to dicts
    readings_list = []
    
    # Add past readings in reverse (oldest first)
    for record in reversed(recent):
        readings_list.append({
            'air_temperature': record.air_temperature,
            'process_temperature': record.process_temperature,
            'rotational_speed': record.rotational_speed,
            'torque': record.torque,
            'tool_wear': record.tool_wear
        })
    
    # Add current reading
    readings_list.append({
        'air_temperature': air_temp,
        'process_temperature': proc_temp,
        'rotational_speed': speed,
        'torque': torque,
        'tool_wear': wear
    })
    
    # Pad with zeros if we don't have 5 readings yet
    while len(readings_list) < 5:
        readings_list.insert(0, {
            'air_temperature': 0,
            'process_temperature': 0,
            'rotational_speed': 0,
            'torque': 0,
            'tool_wear': 0
        })
    
    # Latest reading is the one we're predicting
    latest = readings_list[-1]
    prev1 = readings_list[-2] if len(readings_list) > 1 else readings_list[-1] #picks second last if list size > 1
    
    #creates arrays of just last 5 air_temperature , torque , speed
    prev5_air = [r['air_temperature'] for r in readings_list[-5:]]
    prev5_torque = [r['torque'] for r in readings_list[-5:]]
    prev5_speed = [r['rotational_speed'] for r in readings_list[-5:]]
    
    # For velocity, we need the average of last 3 vs previous 3
    prev3_air = [r['air_temperature'] for r in readings_list[-3:]]
    prev3_wear = [r['tool_wear'] for r in readings_list[-3:]]
    
    # Compute velocity (change in rolling average)
    if len(readings_list) >= 6:
        prev3_prev_air = [r['air_temperature'] for r in readings_list[-6:-3]]
        prev3_prev_wear = [r['tool_wear'] for r in readings_list[-6:-3]]
        air_temp_velocity = np.mean(prev3_air) - np.mean(prev3_prev_air)
        wear_velocity = np.mean(prev3_wear) - np.mean(prev3_prev_wear)
    else:
        air_temp_velocity = 0
        wear_velocity = 0
    
    # Compute features exactly as training script does
    features_dict = {
        # Base features
        'Air temperature [K]': latest['air_temperature'],
        'Process temperature [K]': latest['process_temperature'],
        'Rotational speed [rpm]': latest['rotational_speed'],
        'Torque [Nm]': latest['torque'],
        'Tool wear [min]': latest['tool_wear'],
        
        # Trend features (rate of change from previous reading)
        'air_temp_trend_1': latest['air_temperature'] - prev1['air_temperature'],
        'proc_temp_trend_1': latest['process_temperature'] - prev1['process_temperature'],
        'torque_trend_1': latest['torque'] - prev1['torque'],
        'wear_trend_1': latest['tool_wear'] - prev1['tool_wear'],
        
        # Velocity features (change in the rolling average)
        'air_temp_velocity_3': air_temp_velocity,
        'wear_velocity_3': wear_velocity,
        
        # Rolling statistics (window of last 5)
        'air_temp_rolling_std_5': np.std(prev5_air) if len(prev5_air) > 0 else 0,
        'torque_rolling_max_5': np.max(prev5_torque) if len(prev5_torque) > 0 else 0,
        'speed_rolling_min_5': np.min(prev5_speed) if len(prev5_speed) > 0 else 0,
        
        # Cumulative wear
        'wear_cumulative': latest['tool_wear'],
    }
    
    return features_dict

def dashboard(request):
    # ────── HANDLING A DATA ENTRY (POST) ──────
    if request.method == 'POST' and model is not None:
        try:
            # Extract sensor values from form
            air_temp = float(request.POST.get('air_temp'))
            proc_temp = float(request.POST.get('proc_temp'))
            speed = float(request.POST.get('speed'))
            torque = float(request.POST.get('torque'))
            wear = float(request.POST.get('wear'))
            
            # Compute time-series features
            features_dict = compute_lagged_features(air_temp, proc_temp, speed, torque, wear)
            
            # Build feature vector in the same order as training
            feature_vector = [features_dict[f] for f in feature_names]
            
            # PREDICTION
            prediction_result = model.predict([feature_vector])
            prediction = int(prediction_result[0])
            
            # Get confidence (probability of the predicted class)
            proba = model.predict_proba([feature_vector])[0]
            confidence = float(max(proba)) * 100  # Convert to percentage
            
            # Save to database with all features
            MachineData.objects.create(
                air_temperature=air_temp,
                process_temperature=proc_temp,
                rotational_speed=speed,
                torque=torque,
                tool_wear=wear,
                air_temp_trend_1=features_dict['air_temp_trend_1'],
                proc_temp_trend_1=features_dict['proc_temp_trend_1'],
                torque_trend_1=features_dict['torque_trend_1'],
                wear_trend_1=features_dict['wear_trend_1'],
                air_temp_velocity_3=features_dict['air_temp_velocity_3'],
                wear_velocity_3=features_dict['wear_velocity_3'],
                air_temp_rolling_std_5=features_dict['air_temp_rolling_std_5'],
                torque_rolling_max_5=features_dict['torque_rolling_max_5'],
                speed_rolling_min_5=features_dict['speed_rolling_min_5'],
                wear_cumulative=features_dict['wear_cumulative'],
                prediction=prediction,
                confidence=confidence
            )
            
            return redirect('dashboard')
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return redirect('dashboard')
    
    # ────── HANDLING THE PAGE VIEW (GET) ──────
    readings = MachineData.objects.all().order_by('-timestamp')[:50]
    
    return render(request, 'dashboard.html', {'readings': readings})