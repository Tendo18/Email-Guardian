from django.shortcuts import render
import logging
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from classifier.model_service import ModelService, ModelNotLoadedError
from classifier.models import EmailClassification, FeedbackReport
from classifier.serializers import (
    ClassifyEmailSerializer,
    ClassificationResultSerializer,
    FeedbackSerializer,
    FeedbackResponseSerializer,
)
from classifier.features import extract_urls_from_text
from django.views.decorators.csrf import ensure_csrf_cookie

logger = logging.getLogger(__name__)


# Create your views here.
def get_client_ip(request):
    forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '')


class ClassifyEmailView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = ClassifyEmailSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        service = ModelService.get_instance()

        try:
            result = service.predict(
                subject=data.get('subject', ''),
                body=data['body'],
                sender_email=data.get('sender_email', ''),
                reply_to=data.get('reply_to', ''),
                sender_domain=data.get('sender_domain', ''),
            )
        except ModelNotLoadedError as e:
            return Response(
                {'error': 'model_not_ready', 'message': str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            logger.exception('Prediction failed')
            return Response(
                {'error': 'prediction_failed', 'message': 'Something went wrong during classification.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        
        urls = extract_urls_from_text(f"{data.get('subject', '')} {data['body']}")

        classification = EmailClassification.objects.create(
            subject=data.get('subject', ''),
            body=data['body'],
            sender_email=data.get('sender_email', ''),
            sender_domain=data.get('sender_domain', ''),
            reply_to=data.get('reply_to', ''),
            extracted_urls=urls[:50],
            predicted_label=result['label'],
            confidence_score=result['confidence'],
            label_probabilities=result['probabilities'],
            model_version=result['model_version'],
            ip_address=get_client_ip(request) or None,
        )

        response_data = ClassificationResultSerializer(classification).data

        if result['confidence'] < 0.6:
            response_data['warning'] = f"Low confidence ({result['confidence']:.0%}). Consider reviewing manually."

        return Response(response_data, status=status.HTTP_200_OK)


class FeedbackView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = FeedbackSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        feedback = serializer.save()
        return Response(
            FeedbackResponseSerializer(feedback).data,
            status=status.HTTP_201_CREATED
        )


class ClassificationListView(generics.ListAPIView):
    serializer_class = ClassificationResultSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        qs = EmailClassification.objects.all()
        label = self.request.query_params.get('label')
        if label in ('legitimate', 'spam', 'phishing'):
            qs = qs.filter(predicted_label=label)
        return qs


class ModelStatusView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        service = ModelService.get_instance()
        meta = service.metadata
        return Response({
            'is_loaded': service.is_loaded(),
            'model_type': meta.get('model_type', ''),
            'version': meta.get('version', ''),
            'f1_score': meta.get('f1_score'),
            'accuracy': meta.get('accuracy'),
            'training_samples': meta.get('training_samples'),
        })


@ensure_csrf_cookie
def home(request):
    return render(request, 'classifier/index.html')