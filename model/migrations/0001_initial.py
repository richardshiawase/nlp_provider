# Generated by Django 3.2.5 on 2023-04-05 07:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ItemProvider',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_asuransi', models.CharField(max_length=500)),
                ('nama_provider', models.CharField(max_length=500)),
                ('alamat', models.CharField(max_length=500)),
                ('label_name', models.CharField(max_length=300)),
                ('proba_score', models.CharField(max_length=10)),
                ('count_label_name', models.CharField(max_length=2)),
                ('ri', models.CharField(max_length=2)),
                ('rj', models.CharField(max_length=2)),
                ('nama_alamat', models.CharField(max_length=500)),
                ('alamat_prediction', models.CharField(max_length=500)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='List_Processed_Provider',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_provider', models.CharField(max_length=500)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Provider',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nama_asuransi', models.CharField(max_length=500)),
                ('match_percentage', models.DecimalField(decimal_places=2, max_digits=5)),
                ('status_finish', models.CharField(max_length=8)),
                ('file_location', models.CharField(max_length=1000)),
                ('file_location_result', models.CharField(max_length=1000)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Provider_Model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=30)),
                ('accuracy_score', models.DecimalField(decimal_places=2, max_digits=5)),
                ('model_location', models.CharField(max_length=500)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
