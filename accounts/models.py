
from django.db import models

class user_input(models.Model):
    patient_id= models.CharField(max_length=20,unique=True)
    age = models.IntegerField()
    gender= models.IntegerField()
    nhiss = models.IntegerField()
    mrs = models.IntegerField()
    systolic = models.IntegerField()
    distolic = models.IntegerField()
    glucose = models.IntegerField()
    paralysis = models.IntegerField()
    smoking = models.IntegerField()
    bmi = models.IntegerField()
    cholesterol = models.IntegerField()
    tos = models.IntegerField()
    risk = models.IntegerField()

    #def __str__(self):
       # return user_input.patient_id