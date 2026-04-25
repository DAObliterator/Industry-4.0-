from django.db import models
from django.utils import timezone

class MachineData(models.Model):
    # ────── BASE SENSOR VALUES ──────
    air_temperature = models.FloatField()
    process_temperature = models.FloatField()
    rotational_speed = models.FloatField()
    torque = models.FloatField()
    tool_wear = models.FloatField()
    
    # ────── COMPUTED TIME-SERIES FEATURES ──────
    # These are calculated when prediction happens
    air_temp_trend_1 = models.FloatField(default=0)
    proc_temp_trend_1 = models.FloatField(default=0)
    torque_trend_1 = models.FloatField(default=0)
    wear_trend_1 = models.FloatField(default=0)
    
    air_temp_velocity_3 = models.FloatField(default=0)
    wear_velocity_3 = models.FloatField(default=0)
    
    air_temp_rolling_std_5 = models.FloatField(default=0)
    torque_rolling_max_5 = models.FloatField(default=0)
    speed_rolling_min_5 = models.FloatField(default=0)
    
    wear_cumulative = models.FloatField(default=0)
    
    # ────── PREDICTION & METADATA ──────
    prediction = models.IntegerField()  # 0=HEALTHY, 1=FAILURE
    confidence = models.FloatField(default=0)  # Confidence score (0-1)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
        ]
    
    def __str__(self):
        status = "HEALTHY" if self.prediction == 0 else "FAILURE"
        return f"{self.timestamp} - {status}"