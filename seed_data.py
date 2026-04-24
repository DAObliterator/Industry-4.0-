import os
import django
import pandas as pd
import joblib

# 1. Boilerplate to tell this script how to talk to Django/Supabase
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartline.settings')
django.setup()

from core.models import MachineData

def run_seeder():
    # 2. Load the CSV and the Brain
    df = pd.read_csv('predictive_maintenance.csv')
    model = joblib.load('machine_model.pkl')

    # 3. Take 50 random rows from the 10,000 available
    # .sample(50) is a pandas method that picks random rows
    random_samples = df.sample(50)

    print("Seeding data to Supabase...")

    for index, row in random_samples.iterrows():
        # Map the CSV columns to variables
        air_t = float(row['Air temperature [K]'])
        proc_t = float(row['Process temperature [K]'])
        speed = float(row['Rotational speed [rpm]'])
        torque = float(row['Torque [Nm]'])
        wear = float(row['Tool wear [min]'])

        # Ask the brain for a prediction
        prediction = int(model.predict([[air_t, proc_t, speed, torque, wear]])[0])

        # Save to Supabase via the Django Model (DAO)
        MachineData.objects.create(
            air_temperature=air_t,
            process_temperature=proc_t,
            rotational_speed=speed,
            torque=torque,
            tool_wear=wear,
            prediction=prediction
        )

    print(f"✅ Successfully pushed 50 rows to Supabase!")

if __name__ == "__main__":
    run_seeder()
