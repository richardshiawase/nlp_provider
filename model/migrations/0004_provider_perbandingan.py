# Generated by Django 3.2.5 on 2022-11-21 10:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0003_perbandingan_file_location_result'),
    ]

    operations = [
        migrations.CreateModel(
            name='Provider_Perbandingan',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('perbandingan_id', models.CharField(max_length=2)),
                ('name', models.CharField(max_length=500)),
                ('address', models.CharField(max_length=1000)),
                ('selected', models.CharField(max_length=2)),
            ],
        ),
    ]
