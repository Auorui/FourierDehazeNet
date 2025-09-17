from models.baseline import *


MODEL_CLASSES = {
    'DCP': DCP,
    'DehazeNet': DehazeNet,
    'AODNet': AODNet,
    'GCANet': GCANet,
    'GridDehazeNet': GridDehazeNet,
    'MSBDN_DFF': MSBDN_DFF,
    'MSBDN_RDFF': MSBDN_RDFF,
    'FFANet': FFANet,
    'AECRNet': AECRNet,
    'Dehamer': Dehamer,
    'Restormer': Restormer,
    'FourKDehazing': FourKDehazing,
    'dehazeformer_t': dehazeformer_t,
    'dehazeformer_s': dehazeformer_s,
    'dehazeformer_b': dehazeformer_b,
    'dehazeformer_d': dehazeformer_d,
    'dehazeformer_w': dehazeformer_w,
    'dehazeformer_m': dehazeformer_m,
    'dehazeformer_l': dehazeformer_l,
    'gunet_t': gunet_t,
    'gunet_s': gunet_s,
    'gunet_b': gunet_b,
    'gunet_d': gunet_d,
    'TransRA': TransRA,
    'DEANet': DEANet,
    'MB_TaylorFormer_B': MB_TaylorFormer_B,
    'MB_TaylorFormer_L': MB_TaylorFormer_L,
    'C2PNet': C2PNet,
    'AIDTramsformer': AIDTramsformer,
    'TrinityNet': TrinityNet,
    'ChaIR': ChaIR,
    'MixDehazeNet_t': MixDehazeNet_t,
    'MixDehazeNet_s': MixDehazeNet_s,
    'MixDehazeNet_b': MixDehazeNet_b,
    'MixDehazeNet_l': MixDehazeNet_l,
    'SFHformer_t': SFHformer_t,
    'SFHformer_s': SFHformer_s,
    'SFHformer_m': SFHformer_m,
    'SFHformer_l': SFHformer_l,
    'FourierDehazeNet': FourierDehazeNet,
}

def get_dehaze_networks(name):
    if name in MODEL_CLASSES:
        model = MODEL_CLASSES[name]()
        return model
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")

if __name__ == "__main__":
    import torch
    from pyzjr.nn import summary_2, model_complexity_info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_dehaze_networks('FourKDehazing')
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_2(model, input_size=(3, 256, 256))
    model_complexity_info(model, input_size=(3, 256, 256))



