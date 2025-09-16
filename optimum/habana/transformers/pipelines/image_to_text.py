from transformers.pipelines.image_to_text import ImageToTextPipeline
from optimum.habana.transformers.generation import GaudiGenerationConfig

class GaudiImageToTextPipeline(ImageToTextPipeline):
    _default_generation_config = GaudiGenerationConfig(
        max_new_tokens=512,
    )

