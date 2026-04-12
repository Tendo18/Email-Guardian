from django.db import models

# Create your models here.
class EmailClassification(models.Model):
    LABEL_CHOICES = [
        ('legitimate', 'Legitimate'),
        ('spam', 'Spam'),
        ('phishing', 'Phishing'),
    ]
    
    subject = models.CharField(max_length=255, blank=True, default='')
    body = models.TextField()
    sender_email = models.EmailField(blank=True, null=True, default='')
    sender_domain = models.CharField(max_length=255, blank=True, default='')
    reply_to = models.EmailField(blank=True, default='')
    extracted_urls = models.JSONField(default=list, blank=True)  #
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    
    predicted_label = models.CharField(max_length=20, choices=LABEL_CHOICES)
    confidence_score = models.FloatField()
    label_probabilities = models.JSONField( default=dict)
    model_version = models.CharField(max_length=50)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']  
        
    def __str__(self):
        return f'[{self.predicted_label.upper()}] {self.subject[:60]} ({self.created_at:%Y-%m-%d})'
    
class FeedbackReport(models.Model):
    LABEL_CHOICES = EmailClassification.LABEL_CHOICES
    
    classification = models.OneToOneField(EmailClassification, on_delete=models.CASCADE, related_name='feedback_reports')
    correct_label = models.CharField(max_length=20, choices=EmailClassification.LABEL_CHOICES)
    notes = models.TextField(blank=True, default='')
    comment = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']  
        
    def __str__(self):
        return f'Feedback: {self.classification.predicted_label} → {self.correct_label}'
        
    
    