from django.db import models

class MachineData(models.Model):
    # The UCI dataset uses 0 and 1 for failure, 
    # but we can keep labels for the UI if you prefer.
    STATUS_CHOICES = [
        (0, 'No Failure'),
        (1, 'Machine Failure'),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    
    # These 5 match the UCI CSV exactly
    air_temperature = models.FloatField()      # Air temperature [K]
    process_temperature = models.FloatField()  # Process temperature [K]
    rotational_speed = models.FloatField()     # Rotational speed [rpm]
    torque = models.FloatField()               # Torque [Nm]
    tool_wear = models.FloatField()            # Tool wear [min]
    
    # The "Answer" from the ML Brain
    prediction = models.IntegerField(choices=STATUS_CHOICES)

    def __str__(self):
        status = "FAIL" if self.prediction == 1 else "OK"
        return f"{self.timestamp} - {status}"
