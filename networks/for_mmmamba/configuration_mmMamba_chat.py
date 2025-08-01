import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_mmMamba import mmMambaConfig
from .configuration_mmMamba_embedding import mmMambaEmbeddingConfig

logger = logging.get_logger(__name__)


class mmMambaChatConfig(PretrainedConfig):
    model_type = "mmMamba_chat"
    is_composition = True

    def __init__(
        self,
        embedding_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        normalize_encoder_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if embedding_config is None:
            embedding_config = {}
            logger.info(
                "embedding_config is None. Initializing the VisionConfig with default values."
            )

        if llm_config is None:
            llm_config = {}
            logger.info(
                "llm_config is None. Initializing the Config config with default values (`Config`)."
            )

        self.embedding_config = mmMambaEmbeddingConfig(**embedding_config)
        self.llm_config = mmMambaConfig(**llm_config)

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_encoder_output = normalize_encoder_output

        logger.info(f"vision_select_layer: {self.select_layer}")
        logger.info(f"ps_version: {self.ps_version}")
        logger.info(f"min_dynamic_patch: {self.min_dynamic_patch}")
        logger.info(f"max_dynamic_patch: {self.max_dynamic_patch}")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["embedding_config"] = self.embedding_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["pad2square"] = self.pad2square
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["ps_version"] = self.ps_version
        output["min_dynamic_patch"] = self.min_dynamic_patch
        output["max_dynamic_patch"] = self.max_dynamic_patch
        output["normalize_encoder_output"] = self.normalize_encoder_output

        return output
