
network = {}

#### Detector modules
network['backbone'] = {}
network['neck'] = {}
network['head'] = {}



'''
Below is to build backbone
'''

expand_ratio = 0.5
add_identity = [True, True, True, False]
ssp_layer = [False, False, False, True]
num_blocks = [1, 3, 3, 1]
widen_factor = 0.375
stages = 4


# [[64, 128, 3, True, False], [128, 256, 9, True, False],
#                [256, 512, 9, True, False], [512, 1024, 3, False, True]],
in_channels = [64 * widen_factor, 128 * widen_factor, 256 * widen_factor, 512 * widen_factor]
out_channels = [128 * widen_factor, 256 * widen_factor, 512 * widen_factor, 1024 * widen_factor]

for i in range(len(in_channels)):
    in_channels[i] = int(in_channels[i])
    out_channels[i] = int(out_channels[i])



#### Stem Layer
cur_id = 0
network['backbone']['id'] = [[cur_id, {}]]
network['backbone']['id'][cur_id][1]['type'] = 'Focus'
network['backbone']['id'][cur_id][1]['name'] = 'stem'
network['backbone']['id'][cur_id][1]['in_channel'] = 3
network['backbone']['id'][cur_id][1]['out_channel'] = in_channels[0]
network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
network['backbone']['id'][cur_id][1]['kernel'] = 3
network['backbone']['id'][cur_id][1]['prev'] = [-1]
cur_id += 1

stage_ids = []

#### Stages in Backbone
for i in range(stages):

    ### ConvModule
    network['backbone']['id'].append([cur_id, {}])
    network['backbone']['id'][cur_id][1]['type'] = 'ConvModule'
    network['backbone']['id'][cur_id][1]['in_channel'] = in_channels[i]
    network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i]
    network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
    network['backbone']['id'][cur_id][1]['kernel'] = 3
    network['backbone']['id'][cur_id][1]['stride'] = 2
    network['backbone']['id'][cur_id][1]['prev'] = [cur_id - 1]
    cur_id += 1

    if ssp_layer[i]:
        network['backbone']['id'].append([cur_id, {}])
        network['backbone']['id'][cur_id][1]['type'] = 'SPPBottleneck'
        network['backbone']['id'][cur_id][1]['in_channel'] = out_channels[i]
        network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i]
        network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
        network['backbone']['id'][cur_id][1]['kernel'] = (5, 9, 13)
        network['backbone']['id'][cur_id][1]['prev'] = [cur_id - 1]
        cur_id += 1

    ### CSPLayer --- main conv
    network['backbone']['id'].append([cur_id, {}])
    network['backbone']['id'][cur_id][1]['type'] = 'ConvModule'
    network['backbone']['id'][cur_id][1]['in_channel'] = out_channels[i]
    network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i] * expand_ratio
    network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
    network['backbone']['id'][cur_id][1]['kernel'] = 1
    network['backbone']['id'][cur_id][1]['stride'] = 1
    network['backbone']['id'][cur_id][1]['prev'] = [cur_id - 1]
    main_conv_id = cur_id
    cur_id += 1

    ### CSPLayer --- short conv
    network['backbone']['id'].append([cur_id, {}])
    network['backbone']['id'][cur_id][1]['type'] = 'ConvModule'
    network['backbone']['id'][cur_id][1]['in_channel'] = out_channels[i]
    network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i] * expand_ratio
    network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
    network['backbone']['id'][cur_id][1]['kernel'] = 1
    network['backbone']['id'][cur_id][1]['stride'] = 1
    network['backbone']['id'][cur_id][1]['prev'] = [cur_id - 2]
    short_conv_id = cur_id
    cur_id += 1


    ### CSPLayer --- Bottleneck

    for j in range(num_blocks[i]):
        network['backbone']['id'].append([cur_id, {}])
        network['backbone']['id'][cur_id][1]['type'] = 'ConvModule'
        network['backbone']['id'][cur_id][1]['in_channel'] = out_channels[i] * expand_ratio
        network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i] * expand_ratio * 1.0
        network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
        network['backbone']['id'][cur_id][1]['kernel'] = 1
        network['backbone']['id'][cur_id][1]['stride'] = 1
        if j == 0:
            first_input_id = cur_id - 2
        else:
            first_input_id = cur_id - 1
        network['backbone']['id'][cur_id][1]['prev'] = [first_input_id]
        cur_id += 1

        network['backbone']['id'].append([cur_id, {}])
        network['backbone']['id'][cur_id][1]['type'] = 'ConvModule'
        network['backbone']['id'][cur_id][1]['in_channel'] = out_channels[i] * expand_ratio * 1.0
        network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i] * expand_ratio * 1.0
        network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
        network['backbone']['id'][cur_id][1]['kernel'] = 3
        network['backbone']['id'][cur_id][1]['stride'] = 1
        network['backbone']['id'][cur_id][1]['prev'] = [cur_id - 1]
        cur_id += 1


        if add_identity[i]:
            network['backbone']['id'].append([cur_id, {}])
            network['backbone']['id'][cur_id][1]['type'] = 'add'
            network['backbone']['id'][cur_id][1]['prev'] = [first_input_id, cur_id - 1]
            cur_id += 1



    ### CSPLayer --- final conv

    network['backbone']['id'].append([cur_id, {}])
    network['backbone']['id'][cur_id][1]['type'] = 'concat'
    network['backbone']['id'][cur_id][1]['prev'] = [short_conv_id, cur_id - 1]
    cur_id += 1

    network['backbone']['id'].append([cur_id, {}])
    network['backbone']['id'][cur_id][1]['type'] = 'ConvModule'
    network['backbone']['id'][cur_id][1]['in_channel'] = out_channels[i] * expand_ratio * 2
    network['backbone']['id'][cur_id][1]['out_channel'] = out_channels[i]
    network['backbone']['id'][cur_id][1]['activation'] = 'ReLU'
    network['backbone']['id'][cur_id][1]['kernel'] = 1
    network['backbone']['id'][cur_id][1]['prev'] = [cur_id - 1]
    network['backbone']['id'][cur_id][1]['stride'] = 1

    if i != 0:
        network['backbone']['id'][cur_id][1]['final_out'] = True
    cur_id += 1



# print(network['backbone'])

out_dataflow = {}
for i in range(len(network['backbone']['id'])):
    print(network['backbone']['id'][i])

    for j in range(len(network['backbone']['id'][i][1]['prev'])):
        if network['backbone']['id'][i][1]['prev'][j] < 0:
            pass
        else:
            if network['backbone']['id'][i][1]['prev'][j] not in out_dataflow:
                out_dataflow[network['backbone']['id'][i][1]['prev'][j]] = []
                out_dataflow[network['backbone']['id'][i][1]['prev'][j]].append(network['backbone']['id'][i][0])
            else:
                out_dataflow[network['backbone']['id'][i][1]['prev'][j]].append(network['backbone']['id'][i][0])

# for i in out_dataflow:
#     print(i, out_dataflow[i])



'''
Below is to build neck
'''
in_channels = [96, 192, 384]
out_channels = 96

network['neck']['top_down'] = {}
num_blocks = 1
expand_ratio = 0.5
add_identity = [False, False, False, False]
last_out_dim = 3
bottomup_out_id = [-3]

cur_id = 0
for i in range(len(in_channels) - 1, 0, -1):
    if i == len(in_channels) - 1:
        network['neck']['top_down']['id'] = [[cur_id, {}]]
    else:
        network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'ConvModule'
    network['neck']['top_down']['id'][cur_id][1]['in_channel'] = in_channels[i]
    network['neck']['top_down']['id'][cur_id][1]['out_channel'] = in_channels[i - 1]
    network['neck']['top_down']['id'][cur_id][1]['activation'] = 'ReLU'
    network['neck']['top_down']['id'][cur_id][1]['kernel'] = 1
    network['neck']['top_down']['id'][cur_id][1]['stride'] = 1
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [-1 - i] if i == len(in_channels) - 1 else [cur_id - 1]
    bottomup_out_id[0] = cur_id
    cur_id += 1


    #### upsampling
    network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'upsample'
    network['neck']['top_down']['id'][cur_id][1]['mode'] = 'nearest'
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [cur_id - 1]
    cur_id += 1

    ####concat
    network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'concat'
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [-i, cur_id - 1]
    cur_id += 1

    ### CSPLayer --- main conv
    network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'ConvModule'
    network['neck']['top_down']['id'][cur_id][1]['in_channel'] = in_channels[i - 1] * 2
    network['neck']['top_down']['id'][cur_id][1]['out_channel'] = in_channels[i - 1] * expand_ratio
    network['neck']['top_down']['id'][cur_id][1]['activation'] = 'ReLU'
    network['neck']['top_down']['id'][cur_id][1]['kernel'] = 1
    network['neck']['top_down']['id'][cur_id][1]['stride'] = 1
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [cur_id - 1]
    cur_id += 1

    ### CSPLayer --- short conv
    network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'ConvModule'
    network['neck']['top_down']['id'][cur_id][1]['in_channel'] = in_channels[i - 1] * 2
    network['neck']['top_down']['id'][cur_id][1]['out_channel'] = in_channels[i - 1] * expand_ratio
    network['neck']['top_down']['id'][cur_id][1]['activation'] = 'ReLU'
    network['neck']['top_down']['id'][cur_id][1]['kernel'] = 1
    network['neck']['top_down']['id'][cur_id][1]['stride'] = 1
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [cur_id - 2]
    short_conv_id = cur_id
    cur_id += 1

    ### CSPLayer --- Bottleneck
    for j in range(num_blocks):
        network['neck']['top_down']['id'].append([cur_id, {}])
        network['neck']['top_down']['id'][cur_id][1]['type'] = 'ConvModule'
        network['neck']['top_down']['id'][cur_id][1]['in_channel'] = in_channels[i - 1] * expand_ratio
        network['neck']['top_down']['id'][cur_id][1]['out_channel'] = in_channels[i - 1] * expand_ratio
        network['neck']['top_down']['id'][cur_id][1]['activation'] = 'ReLU'
        network['neck']['top_down']['id'][cur_id][1]['kernel'] = 1
        network['neck']['top_down']['id'][cur_id][1]['stride'] = 1
        if j == 0:
            first_input_id = cur_id - 2
        else:
            first_input_id = cur_id - 1
        network['neck']['top_down']['id'][cur_id][1]['prev'] = [first_input_id]
        cur_id += 1

        network['neck']['top_down']['id'].append([cur_id, {}])
        network['neck']['top_down']['id'][cur_id][1]['type'] = 'ConvModule'
        network['neck']['top_down']['id'][cur_id][1]['in_channel'] = in_channels[i - 1] * expand_ratio
        network['neck']['top_down']['id'][cur_id][1]['out_channel'] = in_channels[i - 1] * expand_ratio
        network['neck']['top_down']['id'][cur_id][1]['activation'] = 'ReLU'
        network['neck']['top_down']['id'][cur_id][1]['kernel'] = 3
        network['neck']['top_down']['id'][cur_id][1]['stride'] = 1
        network['neck']['top_down']['id'][cur_id][1]['prev'] = [cur_id - 1]
        cur_id += 1

        if add_identity[i]:
            network['neck']['top_down']['id'].append([cur_id, {}])
            network['neck']['top_down']['id'][cur_id][1]['type'] = 'add'
            network['neck']['top_down']['id'][cur_id][1]['prev'] = [first_input_id, cur_id - 1]
            cur_id += 1

    ### CSPLayer --- final conv
    network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'concat'
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [short_conv_id, cur_id - 1]
    cur_id += 1

    network['neck']['top_down']['id'].append([cur_id, {}])
    network['neck']['top_down']['id'][cur_id][1]['type'] = 'ConvModule'
    network['neck']['top_down']['id'][cur_id][1]['in_channel'] = in_channels[i - 1] * expand_ratio * 2
    network['neck']['top_down']['id'][cur_id][1]['out_channel'] = in_channels[i - 1]
    network['neck']['top_down']['id'][cur_id][1]['activation'] = 'ReLU'
    network['neck']['top_down']['id'][cur_id][1]['kernel'] = 1
    network['neck']['top_down']['id'][cur_id][1]['stride'] = 1
    network['neck']['top_down']['id'][cur_id][1]['prev'] = [cur_id - 1]
    bottomup_out_id.insert(0, cur_id)
    cur_id += 1


id_here = cur_id
network['neck']['bottom_up'] = {}
outs_id = [bottomup_out_id[0]]



for i in range(len(in_channels) - 1):
    if i == 0:
        network['neck']['bottom_up']['id'] = [[cur_id, {}]]
    else:
        network['neck']['bottom_up']['id'].append([cur_id, {}])
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i]
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['out_channel'] = in_channels[i]
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['kernel'] = 3
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['stride'] = 2
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [outs_id[-1]]
    cur_id += 1

    ####concat
    network['neck']['bottom_up']['id'].append([cur_id, {}])
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'concat'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [bottomup_out_id[i + 1], cur_id - 1]
    cur_id += 1

    ### CSPLayer --- main conv
    network['neck']['bottom_up']['id'].append([cur_id, {}])
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i] * 2
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['out_channel'] = in_channels[i + 1] * expand_ratio
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['kernel'] = 1
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['stride'] = 1
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [cur_id - 1]
    cur_id += 1

    ### CSPLayer --- short conv
    network['neck']['bottom_up']['id'].append([cur_id, {}])
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i] * 2
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['out_channel'] = in_channels[i + 1] * expand_ratio
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['kernel'] = 1
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['stride'] = 1
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [cur_id - 2]
    short_conv_id = cur_id
    cur_id += 1

    for j in range(num_blocks):
        network['neck']['bottom_up']['id'].append([cur_id, {}])
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i + 1] * expand_ratio
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['out_channel'] = in_channels[i + 1] * expand_ratio
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['kernel'] = 1
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['stride'] = 1
        if j == 0:
            first_input_id = cur_id - 2
        else:
            first_input_id = cur_id - 1
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [first_input_id]
        cur_id += 1

        network['neck']['bottom_up']['id'].append([cur_id, {}])
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i + 1] * expand_ratio
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['out_channel'] = in_channels[i + 1] * expand_ratio
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['kernel'] = 3
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['stride'] = 1
        network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [cur_id - 1]
        cur_id += 1

        if add_identity[i]:
            network['neck']['bottom_up']['id'].append([cur_id, {}])
            network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'add'
            network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [first_input_id, cur_id - 1]
            cur_id += 1

    ### CSPLayer --- final conv
    network['neck']['bottom_up']['id'].append([cur_id, {}])
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'concat'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [short_conv_id, cur_id - 1]
    cur_id += 1

    network['neck']['bottom_up']['id'].append([cur_id, {}])
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i + 1] * expand_ratio * 2
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['out_channel'] = in_channels[i + 1]
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['kernel'] = 1
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['stride'] = 1
    network['neck']['bottom_up']['id'][cur_id - id_here][1]['prev'] = [cur_id - 1]
    outs_id.append(cur_id)

    cur_id += 1

id_here = cur_id

###### Final layer of neck
network['neck']['final'] = {}
for i in range(last_out_dim):
    if i == 0:
        network['neck']['final']['id'] = [[cur_id, {}]]
    else:
        network['neck']['final']['id'].append([cur_id, {}])
    network['neck']['final']['id'][cur_id - id_here][1]['type'] = 'ConvModule'
    network['neck']['final']['id'][cur_id - id_here][1]['in_channel'] = in_channels[i]
    network['neck']['final']['id'][cur_id - id_here][1]['out_channel'] = out_channels
    network['neck']['final']['id'][cur_id - id_here][1]['activation'] = 'ReLU'
    network['neck']['final']['id'][cur_id - id_here][1]['kernel'] = 1
    network['neck']['final']['id'][cur_id - id_here][1]['stride'] = 1
    network['neck']['final']['id'][cur_id - id_here][1]['prev'] = [outs_id[i]]
    cur_id += 1

for i in range(len(network['neck']['top_down']['id'])):
    print('top_down', network['neck']['top_down']['id'][i])

for i in range(len(network['neck']['bottom_up']['id'])):
    print('bottom_up', network['neck']['bottom_up']['id'][i])

for i in range(len(network['neck']['final']['id'])):
    print('final', network['neck']['final']['id'][i])


print(bottomup_out_id)
print(outs_id)




'''
Below is to build head
'''
stack_nums = 2
network['head'] = {}
in_channels = 96
feat_channels = 96
cls_out_channels = 80

cur_id = 0

for i in range(3):
    if i == 0:
        network['head']['id'] = [[cur_id, {}]]
    else:
        network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'ConvModule'
    network['head']['id'][cur_id][1]['in_channel'] = in_channels
    network['head']['id'][cur_id][1]['out_channel'] = feat_channels
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 3
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [-1] if i == 0 else [cur_id - 1]
    cur_id += 1

    network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'ConvModule'
    network['head']['id'][cur_id][1]['in_channel'] = feat_channels
    network['head']['id'][cur_id][1]['out_channel'] = feat_channels
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 3
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [cur_id - 1]

    cur_id += 1

cls_id = cur_id - 1

for i in range(3):
    network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'ConvModule'
    network['head']['id'][cur_id][1]['in_channel'] = in_channels
    network['head']['id'][cur_id][1]['out_channel'] = feat_channels
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 3
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [-1] if i == 0 else [cur_id - 1]
    cur_id += 1

    network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'ConvModule'
    network['head']['id'][cur_id][1]['in_channel'] = feat_channels
    network['head']['id'][cur_id][1]['out_channel'] = feat_channels
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 3
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [cur_id - 1]

    cur_id += 1

reg_id = cur_id - 1

for i in range(3):
    network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'Conv'
    network['head']['id'][cur_id][1]['in_channel'] = feat_channels
    network['head']['id'][cur_id][1]['out_channel'] = cls_out_channels
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 1
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [cls_id] if i == 0 else [cur_id - 1]
    cur_id += 1

for i in range(3):
    network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'Conv'
    network['head']['id'][cur_id][1]['in_channel'] = feat_channels
    network['head']['id'][cur_id][1]['out_channel'] = 4
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 1
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [reg_id] if i == 0 else [cur_id - 1]
    cur_id += 1

for i in range(3):
    network['head']['id'].append([cur_id, {}])
    network['head']['id'][cur_id][1]['type'] = 'Conv'
    network['head']['id'][cur_id][1]['in_channel'] = feat_channels
    network['head']['id'][cur_id][1]['out_channel'] = 1
    network['head']['id'][cur_id][1]['activation'] = 'ReLU'
    network['head']['id'][cur_id][1]['kernel'] = 1
    network['head']['id'][cur_id][1]['stride'] = 1
    network['head']['id'][cur_id][1]['prev'] = [reg_id] if i == 0 else [cur_id - 1]
    cur_id += 1


for i in range(len(network['head']['id'])):
    print(network['head']['id'][i])



















