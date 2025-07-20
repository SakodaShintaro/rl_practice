from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import transformers
from torch.nn import CrossEntropyLoss
from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
    TextStreamer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import logging as hf_logging

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from mamba_ssm.utils.generation import (
    modify_logit_for_repetition_penalty,
    sample,
    update_graph_cache,
)

from .configuration_mmMamba_chat import mmMambaChatConfig
from .conversation import get_conv_template
from .modeling_mmMamba import mmMambaForCausalLM
from .modeling_mmMamba_embedding import mmMambaEmbedding

logger = hf_logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    max_new_tokens=None,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    pad_token_id=None,
    do_sample=False,
    teacher_outputs=None,
    vocab_size=None,
    use_cache=False,
    enable_timing=False,
    streamer: Optional[TextStreamer] = None,
    pixel_values=None,
    hd_input_ids=None,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    scores, sequences = [], [input_ids.cpu()]
    if max_new_tokens is not None:
        max_length = (
            sequences[-1].shape[1] + max_new_tokens
        )  # override max_length if max_new_tokens is set

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0

    if not hasattr(model, "_decoding_cache"):
        model._decoding_cache = None

    model._decoding_cache = update_graph_cache(
        model,
        model._decoding_cache,
        batch_size,
        seqlen_og,
        max_length,
    )
    inference_params = model._decoding_cache.inference_params
    inference_params.reset(max_length, batch_size)

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
                return_dict=True,
                pixel_values=pixel_values,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        start.record()
    sequences_cat = input_ids

    while not should_stop(sequences[-1], inference_params):
        torch.cuda.synchronize()
        torch.cuda.reset_max_memory_allocated()
        score = get_logits(sequences[-1].cuda(), inference_params)
        inference_params.seqlen_offset += sequences[-1].shape[1]

        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(score, inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(
                score.clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)

        sequences.append(sampled_tokens.cpu())
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())

    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class MambaGenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        do_sample=False,
        max_length=256,
        max_new_tokens=None,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        if not do_sample:
            top_k = 1
        output = decode(
            input_ids,
            self,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


class mmMambaChatModel(PreTrainedModel):
    config_class = mmMambaChatConfig
    # main_input_name = 'pixel_values'
    _no_split_modules = [
        "InternVisionModel",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
    ]
    _supports_flash_attn_2 = True

    def __init__(self, config: mmMambaChatConfig, embedding_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.embedding_config.image_size
        patch_size = config.embedding_config.patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.use_thumbnail = config.use_thumbnail

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = mmMambaEmbedding(config.embedding_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = mmMambaForCausalLM(config.llm_config)

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.num_samples = 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
        query=None,
        hd_input_ids=None,
        hd_input_embeds=None,
        hd_labels=None,
        hd_loss_weight=None,
        inference_params=None,
        num_last_tokens: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is not None or input_ids.shape[0] > 1:
            if image_flags is not None:
                # image_flags = image_flags.squeeze(-1)
                pixel_values = pixel_values[image_flags == 1]
            if pixel_values == []:
                pixel_values = None
            if getattr(self.embedding_model.config, "pixel_shuffle_loc", None) in ["post"]:
                assert hd_input_ids is not None, (
                    "hd_input_ids is required for pixel_shuffle_loc=post"
                )
                embedding_input_ids = hd_input_ids
            else:
                embedding_input_ids = input_ids
            image_embeds, input_embeds = self.embedding_model(
                input_ids=embedding_input_ids,
                pixel_values=pixel_values,
                use_cache=use_cache,
                return_dict=return_dict,
                inference_params=inference_params,
            )

            B, N = embedding_input_ids.shape
            image_batch_size = pixel_values.shape[0] if pixel_values is not None else 0
            C = image_embeds.shape[-1]
            input_embeds = input_embeds.reshape(B * N, C)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                # print(f'dynamic ViT batch size: {image_batch_size}, images per sample: {image_batch_size / B}, dynamic token length: {N}')
                if statistics is not None:
                    num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                    self.num_samples += num_samples
                    print(
                        f"total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}"
                    )

            if image_batch_size != 0:
                if getattr(self.embedding_model.config, "pixel_shuffle_loc", None) == "post":
                    B, N = input_ids.shape
                    llm_input_embeds = torch.zeros(
                        input_ids.shape[1], C, device=input_ids.device, dtype=input_embeds.dtype
                    )
                    llm_selected = input_ids.flatten() == self.img_context_token_id
                    hd_llm_selected = hd_input_ids.flatten() == self.img_context_token_id
                    llm_input_embeds[~llm_selected] = input_embeds[~hd_llm_selected]
                    llm_input_embeds[llm_selected] = image_embeds.reshape(-1, C)
                    input_embeds = llm_input_embeds

            input_embeds = input_embeds.reshape(B, N, C)

        else:
            input_embeds = self.embedding_model.get_input_embeddings(input_ids)
            hd_input_ids = input_ids
            hd_input_embeds = input_embeds
            next_past_key_values = []
            if getattr(self.embedding_model.config, "pixel_shuffle_loc", None) in ["post"]:
                embedding_input_embeds = hd_input_embeds
            else:
                embedding_input_embeds = input_embeds
            for layer_idx, layer_module in enumerate(self.embedding_model.encoder):
                outputs = layer_module(
                    hidden_states=embedding_input_embeds,
                    use_cache=use_cache,
                    return_dict=return_dict,
                    inference_params=inference_params,
                )
                embedding_input_embeds = outputs[0]

            input_embeds = embedding_input_embeds

        if self.config.normalize_encoder_output:
            input_embeds = input_embeds / input_embeds.norm(dim=-1, keepdim=True)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        next_past_key_values = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print("Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.")

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values, input_ids=input_ids, **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        hd_query = deepcopy(query)
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            hd_image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN
                * int(self.num_image_token // self.downsample_ratio**2)
                * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            hd_query = hd_query.replace("<image>", hd_image_tokens, 1)
            # print(hd_query)

        model_inputs = tokenizer(query, return_tensors="pt")
        hd_model_inputs = tokenizer(hd_query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        hd_input_ids = hd_model_inputs["input_ids"].cuda()

        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            hd_input_ids=hd_input_ids,
            **generation_config,
        )
        generation_output = generation_output[:, input_ids.shape[1] :]

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
            if verbose:
                print(query_to_print, response)
            return response

    def generate(self, *args, **kwargs):
        return MambaGenerationMixin.generate(self, *args, **kwargs)

    def allocate_inference_cache(self, *args, **kwargs):
        dict1 = self.embedding_model.allocate_inference_cache(*args, **kwargs)
        dict2 = self.language_model.allocate_inference_cache(*args, **kwargs)
        return {**dict1, **dict2}
