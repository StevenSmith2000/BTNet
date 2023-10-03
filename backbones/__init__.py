from .iresnet_multi_reso_distill import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf

def get_model(name, resolution, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(True, resolution, **kwargs)
    elif name == "r34":
        return iresnet34(True, resolution, **kwargs)
    elif name == "r50":
        return iresnet50(True, resolution, **kwargs)
    elif name == "r100":
        return iresnet100(True, resolution, **kwargs)
    elif name == "r200":
        return iresnet200(True, resolution, **kwargs)
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)
    else:
        raise ValueError()

if __name__ == "__main__":
    model = iresnet50(True,7)
    for name, param in model.named_parameters():
        print(name)
