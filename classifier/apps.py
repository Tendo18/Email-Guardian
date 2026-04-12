from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"

    def ready(self):
        import sys
        if "migrate" in sys.argv or "makemigrations" in sys.argv:
            return
        try:
            from classifier.model_service import ModelService
            ModelService.get_instance().load()
        except Exception:
            pass