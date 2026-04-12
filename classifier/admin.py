from django.contrib import admin
from classifier.models import EmailClassification, FeedbackReport

# Register your models here.
@admin.register(EmailClassification)
class EmailClassificationAdmin(admin.ModelAdmin):
    list_display = ['id', 'short_subject', 'predicted_label', 'confidence_score', 'sender_email', 'created_at']
    list_filter = ['predicted_label', 'created_at']
    search_fields = ['subject', 'sender_email']
    readonly_fields = ['created_at', 'label_probabilities', 'extracted_urls']
    ordering = ['-created_at']

    def short_subject(self, obj):
        return obj.subject[:60] or '(no subject)'
    short_subject.short_description = 'Subject'


@admin.register(FeedbackReport)
class FeedbackReportAdmin(admin.ModelAdmin):
    list_display = ['id', 'classification', 'correct_label', 'created_at']
    list_filter = ['correct_label', 'created_at']
    readonly_fields = ['created_at']
    ordering = ['-created_at']