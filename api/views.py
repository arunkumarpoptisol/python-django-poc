from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import ItemSerializer
from polls.models import Question
from django.http import HttpResponse

@api_view(['GET'])
def getData(request):
    try:
        QuestionsData = Question.objects.all()
        serializerData=ItemSerializer(QuestionsData,many=True)
        return Response(serializerData.data)
    except:
        return HttpResponse('404')
    
@api_view(['POST'])
def addData(request):
    serialized=ItemSerializer(data=request.data)
    if serialized.is_valid():
        serialized.save()
    return Response(serialized.data)