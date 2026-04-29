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

        import threading

        def load_or_train():
            try:
                from classifier.model_service import ModelService
                from django.conf import settings

                service = ModelService.get_instance()
                loaded = service.load()

                if not loaded:
                    logger.info("No model found — training synthetic model in background...")
                    from classifier.training import train
                    train(output_dir=str(settings.MODELS_DIR))
                    service.reload()
                    logger.info("Background training complete — model is ready.")

            except Exception as e:
                logger.warning(f"Model loading/training failed: {e}")

        thread = threading.Thread(target=load_or_train, daemon=True)
        thread.start()