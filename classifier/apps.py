from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ClassifierConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"

    def ready(self):
        import sys
        if "migrate" in sys.argv or "makemigrations" in sys.argv:
            return

        try:
            from classifier.model_service import ModelService
            service = ModelService.get_instance()
            loaded = service.load()

            if not loaded:
                logger.info("No model found — training synthetic model at startup...")
                from classifier.training import train
                from django.conf import settings
                train(output_dir=str(settings.MODELS_DIR))
                service.reload()
                logger.info("Startup training complete.")

        except Exception as e:
            logger.warning(f"Startup model loading failed: {e}")