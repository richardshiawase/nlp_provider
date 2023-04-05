# Generated by Django 3.2.5 on 2023-04-04 04:48

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('model', '0003_list_processed_provider'),
    ]

    operations = [
        migrations.AddField(
            model_name='itemprovider',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='itemprovider',
            name='nama_alamat',
            field=models.CharField(default='-', max_length=500),
            preserve_default=False,
        ),
    ]
