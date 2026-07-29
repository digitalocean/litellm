import pytest

from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

GPT5_6_PLUS_MODELS = [
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "openai/gpt-5.6-sol",
]

GPT5_PRE_5_6_MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-4o",
]


class TestOpenAIGPT5ConfigIsModelGpt56PlusModel:
    @pytest.mark.parametrize("model", GPT5_6_PLUS_MODELS)
    def test_is_model_gpt_5_6_plus_model_true(self, model):
        assert OpenAIGPT5Config.is_model_gpt_5_6_plus_model(model) is True

    @pytest.mark.parametrize("model", GPT5_PRE_5_6_MODELS)
    def test_is_model_gpt_5_6_plus_model_false(self, model):
        assert OpenAIGPT5Config.is_model_gpt_5_6_plus_model(model) is False
