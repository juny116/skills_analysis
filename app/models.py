from django.db import models

CHOICES = [
    ('CONCAT', 'concat'),
    ('AVG', 'average'),
]


class MaskString(models.Model):
    mask = models.CharField(max_length=100, blank=True)


class WeightSetup(models.Model):
    name = models.DecimalField(max_digits=5, decimal_places=3)
    description = models.DecimalField(max_digits=5, decimal_places=3)
    class_1 = models.DecimalField(max_digits=5, decimal_places=3)
    class_2 = models.DecimalField(max_digits=5, decimal_places=3)
    abilities = models.DecimalField(max_digits=5, decimal_places=3)
    objective = models.DecimalField(max_digits=5, decimal_places=3)
    type = models.CharField(
        max_length=20,
        choices=CHOICES,
        default='CONCAT',
    )

class Job(models.Model):
    id = models.CharField(max_length=30, primary_key=True)
    name = models.CharField(max_length=30)
    description = models.CharField(max_length=300)
    class_1 = models.CharField(max_length=30)
    class_2 = models.CharField(max_length=30)
    abilities = models.CharField(max_length=300)
    objective = models.CharField(max_length=300)
    q_digit = models.CharField(max_length=30)