# Generated by Django 3.2.5 on 2023-04-05 02:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0004_auto_20230404_1148'),
    ]

    operations = [
        migrations.AddField(
            model_name='itemprovider',
            name='alamat_prediction',
            field=models.CharField(default='-', max_length=500),
            preserve_default=False,
        ),
    ]
