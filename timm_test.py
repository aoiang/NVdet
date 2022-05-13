import timm

m = timm.create_model('convnext_small', pretrained=True)

# for i, layer in enumerate(m.stages):
#     print(i, layer)

# for i, layer in enumerate(m.stages):
#     for j, l in enumerate(layer):
#         print(j, l)


for name, param in m.named_parameters():
    if param.requires_grad:
        print(name)