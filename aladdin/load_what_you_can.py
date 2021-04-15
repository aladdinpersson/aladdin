import torch

def load_what_you_can(pretrained, model, load_running_averages=True, verbose=True):
    is_dict = isinstance(pretrained, dict)
    dict_pretrained = pretrained if is_dict else pretrained.state_dict()
    dict_not_pretrained = model.state_dict()
    iter_dict = dict_not_pretrained.copy()

    for name, param in dict_pretrained.items():
        if "running" in name and not load_running_averages:
            print(f"Skipping running average metric of layer: {name}")
            continue
        
        if verbose: print(f"Current: {name}")
        same_name, same_size = None, None

        for name2, param2 in iter_dict.items():
            if name == name2 and same_name is None:
                same_name = name

            if param.size() == param2.size() and same_size is None:
                same_size = name2

        if same_name and dict_not_pretrained[same_name].size() == dict_not_pretrained[same_name].size():
            if verbose: print(f"Successfully loaded: {same_name}")
            dict_not_pretrained[same_name].data.copy_(dict_pretrained[same_name])
            del iter_dict[same_name]

        elif same_size:
            if verbose: print(f"Successfully loaded: {name}")
            dict_not_pretrained[same_size].data.copy_(dict_pretrained[name])
            del iter_dict[same_size]
        else:
            if verbose: print(f"Did not load: {name}")

    model.load_state_dict(dict_not_pretrained)





