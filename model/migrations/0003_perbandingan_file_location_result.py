# Generated by Django 3.2.5 on 2022-11-07 04:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0002_perbandingan'),
    ]

    operations = [
        migrations.AddField(
            model_name='perbandingan',
            name='file_location_result',
            field=models.CharField(default='-', max_length=1000),
            preserve_default=False,
        ),
    ]
