from rest_framework import serializers
from polls.models import Question

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields='__all__'