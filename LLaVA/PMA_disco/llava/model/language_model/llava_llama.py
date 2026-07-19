#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.training = False
        self.cur_task = 0
        self.prog_signal = False
        self.lora_id = self.cur_task
        self.expert_num = 8
        self.global_text_feature = nn.ParameterList(
                [nn.Parameter(torch.zeros(768, dtype=torch.bfloat16)) for _ in range(8)]
            )
        self.global_image_feature = nn.ParameterList(
                [nn.Parameter(torch.zeros(768, dtype=torch.bfloat16)) for _ in range(8)]
            )
        self.image_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(8)]
            )
        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(8)]
            )

    def set_cur_task(self, cur_task, expert_num, prog_signal):
        self.cur_task = cur_task
        self.expert_num = expert_num
        self.prog_signal = prog_signal

    def set_boundary_for_save(self):
        for name, param in self.image_boundary.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.global_image_feature.named_parameters():
            param.requires_grad = True
        
        for name, param in self.global_text_feature.named_parameters():
            param.requires_grad = True

    def get_model(self):
        return self.model

    def set_eval(self, num_task):
        self.expert_num = num_task

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                loss_rec,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        if loss_rec is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            if self.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not torch.is_tensor(loss_rec):
                loss_rec = torch.tensor(loss_rec, device=loss.device, dtype=loss.dtype)
            else:
                loss_rec = loss_rec.to(device=loss.device, dtype=loss.dtype)
            import pdb
            # pdb.set_trace()
            loss += loss_rec
            # print("loss: "+str(loss))
            # print("loss_rec: "+str(loss_rec))

            # #########################
            # loss = None
            # ce_loss = None

            # if labels is not None:
            #     # Shift so that tokens < n predict n
            #     shift_logits = logits[..., :-1, :].contiguous()
            #     shift_labels = labels[..., 1:].contiguous()

            #     # ===== DEBUG: 监督信号是否有效 =====
            #     # 注意：LLaVA 通常用 -100 mask 掉非 assistant 的 token
            #     valid = (shift_labels != -100).sum().item()
            #     total = shift_labels.numel()
            #     # 只打印前 50 step 或者一旦 valid=0 就打印
            #     step = getattr(self, "_dbg_step", 0)
            #     if step < 50 or valid == 0:
            #         has_images = images is not None
            #         print(f"[DBG forward] step={step} valid_labels={valid}/{total} has_images={has_images}")

            #     # ===== 正常 CE loss 计算（ignore -100！）=====
            #     loss_fct = CrossEntropyLoss(ignore_index=-100)
            #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
            #     shift_labels = shift_labels.view(-1).to(shift_logits.device)

            #     ce_loss = loss_fct(shift_logits, shift_labels)
            #     loss = ce_loss

            # # ===== DEBUG: loss_rec / 数值稳定性 =====
            # if not torch.is_tensor(loss_rec):
            #     # loss 可能为 None（例如 labels=None），这里要防一下
            #     if loss is None:
            #         # 没有 CE loss 的情况下你还要加 loss_rec？那就需要明确你的训练目标
            #         loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            #     loss_rec_t = torch.tensor(loss_rec, device=loss.device, dtype=loss.dtype)
            # else:
            #     if loss is None:
            #         loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            #     loss_rec_t = loss_rec.to(device=loss.device, dtype=loss.dtype)

            # # 避免 loss_rec 是 nan/inf 把整个 loss 搞成 nan/inf（或奇怪行为）
            # if not torch.isfinite(loss_rec_t).all():
            #     print("[DBG forward] loss_rec is non-finite:", loss_rec_t)

            # loss = loss + loss_rec_t

            # # 打印 loss 组成（前 50 step 或遇到异常）
            # step = getattr(self, "_dbg_step", 0)
            # if step < 50 or (labels is not None and valid == 0) or (not torch.isfinite(loss)):
            #     ce_val = float(ce_loss.detach().cpu()) if ce_loss is not None else None
            #     rec_val = float(loss_rec_t.detach().cpu()) if torch.is_tensor(loss_rec_t) else None
            #     print(f"[DBG forward] step={step} ce_loss={ce_val} rec_loss={rec_val} total_loss={float(loss.detach().cpu())}")

            # self._dbg_step = step + 1
            # ######################




            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
