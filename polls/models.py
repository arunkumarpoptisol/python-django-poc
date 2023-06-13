from django.db import models

# Create your models here.
class Question(models.Model):
    name=models.CharField(max_length=200)
    date=models.DateField('date')

    