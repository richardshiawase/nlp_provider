# Generated by Django 3.2.5 on 2023-04-18 06:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0003_itemprovider_golden_record'),
    ]

    operations = [
        migrations.AddField(
            model_name='goldenrecordmatch',
            name='alamat',
            field=models.CharField(default='-', max_length=500),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='goldenrecordmatch',
            name='nama_provider',
            field=models.CharField(default='-', max_length=500),
            preserve_default=False,
        ),
    ]
