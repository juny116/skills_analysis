# Generated by Django 4.1.2 on 2022-10-25 06:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_delete_abilitytype_weightsetup_type'),
    ]

    operations = [
        migrations.RenameField(
            model_name='weightsetup',
            old_name='abilities',
            new_name='abilities_avg',
        ),
        migrations.RemoveField(
            model_name='weightsetup',
            name='type',
        ),
        migrations.AddField(
            model_name='weightsetup',
            name='abilities_concat',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=5),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='maskstring',
            name='mask',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
