
must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]

def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    return ret
