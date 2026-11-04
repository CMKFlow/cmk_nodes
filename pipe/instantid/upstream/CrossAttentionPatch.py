import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
from .utils import tensor_to_size


CMK_INSTANTID_AREAS = "cmk_instantid_areas"
CMK_INSTANTID_FULL_DIMS = "cmk_instantid_full_dims"


def crop_spatial_tensor(tensor, area, full_dims):
    if area is None:
        return tensor
    if len(area) != 4 or len(full_dims) != 2:
        raise RuntimeError(f"InstantID requires a 2D conditioning area, got {area!r}")

    height, width, y, x = (int(value) for value in area)
    full_height, full_width = (int(value) for value in full_dims)
    if height <= 0 or width <= 0 or full_height <= 0 or full_width <= 0:
        raise RuntimeError(f"InstantID received an invalid conditioning area {area!r}")
    if y < 0 or x < 0 or y + height > full_height or x + width > full_width:
        raise RuntimeError(
            f"InstantID conditioning area {area!r} is outside full latent shape {tuple(full_dims)!r}"
        )

    tensor_height, tensor_width = tensor.shape[-2:]
    y0 = round(y * tensor_height / full_height)
    y1 = round((y + height) * tensor_height / full_height)
    x0 = round(x * tensor_width / full_width)
    x1 = round((x + width) * tensor_width / full_width)
    y0 = min(tensor_height - 1, max(0, y0))
    x0 = min(tensor_width - 1, max(0, x0))
    y1 = min(tensor_height, max(y0 + 1, y1))
    x1 = min(tensor_width, max(x0 + 1, x1))
    return tensor[..., y0:y1, x0:x1]


def resolved_areas_by_uuid(conds, x_in):
    dims = tuple(x_in.shape[2:])
    areas = {}
    for cond_group in conds:
        for cond in cond_group:
            cond_uuid = cond.get("uuid")
            if cond_uuid is None:
                continue
            area = cond.get("area")
            if area is None:
                areas[cond_uuid] = None
                continue
            if len(area) != len(dims) * 2 or not all(isinstance(value, int) for value in area):
                raise RuntimeError(f"InstantID requires a resolved integer conditioning area, got {area!r}")
            area = list(area)
            for index in range(len(dims)):
                area[index] = min(x_in.shape[index + 2] - area[len(dims) + index], area[index])
            areas[cond_uuid] = tuple(area)
    return areas


def instantid_calc_cond_batch(executor, model, conds, x_in, timestep, model_options):
    areas = resolved_areas_by_uuid(conds, x_in)
    full_dims = tuple(x_in.shape[2:])
    controls = {}

    for cond_group in conds:
        for cond in cond_group:
            control = cond.get("control")
            full_hint = getattr(control, "_cmk_instantid_full_hint", None)
            if full_hint is None:
                continue
            cond_uuid = cond.get("uuid")
            if cond_uuid not in areas:
                raise RuntimeError(f"InstantID is missing conditioning area information for UUID {cond_uuid}")
            area = areas[cond_uuid]
            previous_area = controls.get(control, area)
            if control in controls and previous_area != area:
                raise RuntimeError("InstantID ControlNet instance was shared across different conditioning areas")
            controls[control] = area
            area_key = (tuple(full_dims), None if area is None else tuple(area))
            if getattr(control, "_cmk_instantid_active_area", object()) != area_key:
                control.cond_hint_original = crop_spatial_tensor(full_hint, area, full_dims)
                control.cond_hint = None
                control._cmk_instantid_active_area = area_key

    model_options = model_options.copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    transformer_options[CMK_INSTANTID_AREAS] = areas
    transformer_options[CMK_INSTANTID_FULL_DIMS] = full_dims
    model_options["transformer_options"] = transformer_options
    return executor(model, conds, x_in, timestep, model_options)


def neutral_keypoint_hint(image, image_kps):
    return torch.zeros_like(image_kps) if image_kps is not None else torch.zeros_like(image)

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]
    
    def add(self, callback, **kwargs):          
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])
        
        return out.to(dtype=dtype)

def instantid_attention(out, q, k, v, extra_options, module_key='', ipadapter=None, weight=1.0, cond=None, cond_alt=None, uncond=None, weight_type="linear", mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False, embeds_scaling='V only', **kwargs):
    dtype = q.dtype
    cond_or_uncond = extra_options["cond_or_uncond"]
    block_type = extra_options["block"][0]
    #block_id = extra_options["block"][1]
    t_idx = extra_options["transformer_index"]
    layers = 11 if '101_to_k_ip' in ipadapter.ip_layers.to_kvs else 16
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    # extra options for AnimateDiff
    ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

    b = q.shape[0]
    seq_len = q.shape[1]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    if weight_type == 'ease in':
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == 'ease out':
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == 'ease in-out':
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'reverse in-out':
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'weak input' and block_type == 'input':
        weight = weight * 0.2
    elif weight_type == 'weak middle' and block_type == 'middle':
        weight = weight * 0.2
    elif weight_type == 'weak output' and block_type == 'output':
        weight = weight * 0.2
    elif weight_type == 'strong middle' and (block_type == 'input' or block_type == 'output'):
        weight = weight * 0.2
    elif isinstance(weight, dict):
        if t_idx not in weight:
            return 0

        weight = weight[t_idx]

        if cond_alt is not None and t_idx in cond_alt:
            cond = cond_alt[t_idx]
            del cond_alt

    if unfold_batch:
        # Check AnimateDiff context window
        if ad_params is not None and ad_params["sub_idxs"] is not None:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, ad_params["full_length"])
                weight = torch.Tensor(weight[ad_params["sub_idxs"]])
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            # if image length matches or exceeds full_length get sub_idx images
            if cond.shape[0] >= ad_params["full_length"]:
                cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
            # otherwise get sub_idxs images
            else:
                cond = tensor_to_size(cond, ad_params["full_length"])
                uncond = tensor_to_size(uncond, ad_params["full_length"])
                cond = cond[ad_params["sub_idxs"]]
                uncond = uncond[ad_params["sub_idxs"]]
        else:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, batch_prompt)
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            cond = tensor_to_size(cond, batch_prompt)
            uncond = tensor_to_size(uncond, batch_prompt)

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond)
    else:
        # TODO: should we always convert the weights to a tensor?
        if isinstance(weight, torch.Tensor):
            weight = tensor_to_size(weight, batch_prompt)
            if torch.all(weight == 0):
                return 0
            weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
        elif weight == 0:
            return 0

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)

    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)
    
    if embeds_scaling == 'K+mean(V) w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
        ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        del ip_v_mean
    elif embeds_scaling == 'K+V w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    elif embeds_scaling == 'K+V':
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    else:
        #ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        out_ip = out_ip * weight # I'm doing this to get the same results as before

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        # check if using AnimateDiff and sliding context window
        if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
            # if mask length matches or exceeds full_length, get sub_idx masks
            if mask.shape[0] >= ad_params["full_length"]:
                mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            else:
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                mask = tensor_to_size(mask, ad_params["full_length"])
                mask = mask[ad_params["sub_idxs"]]
            mask = mask.repeat(len(cond_or_uncond), 1, 1)
        else:
            uuids = extra_options.get("uuids")
            areas = extra_options.get(CMK_INSTANTID_AREAS)
            full_dims = extra_options.get(CMK_INSTANTID_FULL_DIMS)
            if uuids and areas is not None and full_dims is not None:
                mask_chunks = []
                for cond_uuid in uuids:
                    if cond_uuid not in areas:
                        raise RuntimeError(
                            f"InstantID is missing conditioning area information for UUID {cond_uuid}"
                        )
                    area_mask = crop_spatial_tensor(mask, areas[cond_uuid], full_dims)
                    area_mask = F.interpolate(
                        area_mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear"
                    ).squeeze(1)
                    mask_chunks.append(tensor_to_size(area_mask, batch_prompt))
                mask = torch.cat(mask_chunks, dim=0)
            else:
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                mask = tensor_to_size(mask, batch_prompt)
                mask = mask.repeat(len(cond_or_uncond), 1, 1)

        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    #out = out + out_ip

    return out_ip.to(dtype=dtype)
