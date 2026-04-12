from rest_framework import serializers
from classifier.models import EmailClassification, FeedbackReport


class ClassifyEmailSerializer(serializers.Serializer):
    subject = serializers.CharField(required=False, default='', allow_blank=True)
    body = serializers.CharField(required=True)
    sender_email = serializers.EmailField(required=False, default='', allow_blank=True)
    sender_domain = serializers.CharField(required=False, default='', allow_blank=True)
    reply_to = serializers.EmailField(required=False, default='', allow_blank=True)

    def validate_body(self, value):
        if len(value.strip()) < 5:
            raise serializers.ValidationError('Email body is too short to classify.')
        return value


class ClassificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmailClassification
        fields = [
            'id',
            'subject',
            'sender_email',
            'predicted_label',
            'confidence_score',
            'label_probabilities',
            'model_version',
            'created_at',
        ]
        read_only_fields = fields


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeedbackReport
        fields = ['classification', 'correct_label', 'notes']

    def validate(self, attrs):
        classification = attrs['classification']
        correct_label = attrs['correct_label']

        if classification.predicted_label == correct_label:
            raise serializers.ValidationError({
                'correct_label': 'The model already predicted this label. Only flag misclassified emails.'
            })

        if FeedbackReport.objects.filter(classification=classification).exists():
            raise serializers.ValidationError({
                'classification': 'Feedback already submitted for this email.'
            })

        return attrs


class FeedbackResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeedbackReport
        fields = ['id', 'classification', 'correct_label', 'notes', 'created_at']
        read_only_fields = fields