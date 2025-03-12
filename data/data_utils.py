import torch
from data.prefix_instruction import degradation_list


def check_item_graph200k(data, image_type_list):
    valid = True

    for image_type in image_type_list:
        if not valid:
            return valid
        if image_type in [
            "target", 
            "mask", "bbox", 
            "canny", "depth", "hed", "normal", "openpose", "mlsd", 
            "sam2_mask", "uniformer", 
            "DepthEdit", "FillEdit", "ReduxStyle", "InstantStyle"]:
            if image_type not in data["condition"]:
                valid = False
        elif image_type in ["foreground", "background"]:
            if "foreground" not in data["condition"]:
                valid = False
        elif "reference" == image_type:
            if data["quality_assessment"] is not None:
                if data["quality_assessment"]["objectConsistency"] < 3:
                    valid = False
            else:
                valid = False
        elif image_type in degradation_list:
            valid = True
        else:
            print(image_type)
            raise NotImplementedError()
    return valid


def dataloader_collate_fn(samples):
    group_names = [x[0] for x in samples]
    image = [x[1] for x in samples]
    prompt = [x[2] for x in samples]
    text_emb = [x[3] for x in samples]
    grid_shape = [x[4] for x in samples]
    return group_names, image, prompt, text_emb, grid_shape


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()
