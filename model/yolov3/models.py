import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/bo/research/realtime-action-detection')
from model.yolov3.utils.google_utils import *
from model.yolov3.utils.parse_config import *
from model.yolov3.utils.utils import *

def create_modules(module_defs, img_size, arc):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())

        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list
                                nc=int(mdef['classes']),  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1 or 2
                                arc=arc)  # yolo architecture

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.arc = arc

    def forward(self, p, img_size, var=None):
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 87, 13, 13) -- > (bs, 3, 13, 13, 29)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.permute(0, 2, 3, 1).contiguous()  # prediction

        return p

        # else:  # inference
        #     # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
        #     io = p.clone()  # inference output
        #     io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
        #     io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        #     # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
        #     io[..., :4] *= self.stride

        #     if 'default' in self.arc:  # seperate obj and cls
        #         torch.sigmoid_(io[..., 4])
        #     elif 'BCE' in self.arc:  # unified BCE (80 classes)
        #         torch.sigmoid_(io[..., 5:])
        #         io[..., 4] = 1
        #     elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
        #         io[..., 4:] = F.softmax(io[..., 4:], dim=4)
        #         # io[..., 4] = 1

        #     if self.nc == 1:
        #         io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

        #     # reshape from [1, 3, 13, 13, 85] to [1, 507, 84], remove obj_conf
        #     return io.view(bs, -1, self.no), p


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)
        self.priors = []

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

        self.softmax = nn.Softmax(dim=1).cuda()
        
    def forward(self, x, var=None):
        batch_size = x.shape[0]
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []
        priors = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                tmp = module(x, img_size).view(batch_size,-1,29)
                output.append(tmp)
                if len(self.priors) == 0:
                    size = torch.Tensor([img_size[0],img_size[1]]).cuda()
                    grid_xy = (module.grid_xy.view(-1,2)) * module.stride/size
                    xy = grid_xy.repeat([1,len(module.anchor_vec)]).view(-1,2)
                    wh = module.anchor_vec.repeat([len(grid_xy),1]) * module.stride/size
                    prior = torch.cat((xy,wh),1)
                    priors.append(prior)
            layer_outputs.append(x if i in self.routs else [])

        if len(self.priors) == 0:
            self.priors = torch.cat(priors,0)
        output = torch.cat(output,1)

        loc_preds = output[...,:4]
        cls_preds = output[...,4:]
        return loc_preds, cls_preds, self.priors

def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

if __name__ == '__main__':
    model = Darknet('cfg/yolov3-spp.cfg', arc='default')
    model = model.cuda()
    image = torch.randn(4,3,512,1024).cuda()

    model.train()
    loc_preds, cls_preds, priors = model(image)
    print(loc_preds.shape,cls_preds.shape)
    print(priors.shape)

    # model.eval()
    # inf_out, train_out = model(image)
    # for i in inf_out:
    #     print(i.shape)

    # for t in train_out:
    #     print(t.shape)

    # def xywh2xyxy(x):
    #     # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    #     y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    #     y[:, 0] = x[:, 0] - x[:, 2] / 2
    #     y[:, 1] = x[:, 1] - x[:, 3] / 2
    #     y[:, 2] = x[:, 0] + x[:, 2] / 2
    #     y[:, 3] = x[:, 1] + x[:, 3] / 2
    #     return y

    # min_wh, max_wh = 2, 4096
    # for image_i, pred in enumerate(inf_out):
    #     pred = pred[(pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1)]
    #     pred = pred[pred[:, 4] > 0.001]

    #     if not torch.isfinite(pred).all():
    #         pred = pred[torch.isfinite(pred).all(1)]

    #     loc_data = xywh2xyxy(pred[:,:4])
    #     conf_preds = pred[:,5:]

    # change targets format
    # image_id, x,y,w,h, label

    # use different loss

