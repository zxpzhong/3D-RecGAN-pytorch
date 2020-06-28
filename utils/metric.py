import torch

def IOU_metric(input_,GT,threshold=0.5):
    '''
    input_ : L*H*W*1
    GT : L*H*W*1
    threshold : float number
    return : IOU metric
    '''
    # (input_ > t & GT > t) / (input_ > t | GT > t)
    return torch.sum((input_ > threshold) * (GT > threshold)) / torch.sum((input_ > threshold) + (GT > threshold)).float()

def cross_entropy(a, y):
    
    # y [0,1]
    # print(torch.log(a+1e-8))
    # print(torch.log(1-a+1e-8))
    # print(-y*torch.log(a+1e-8))
    # print((1-y)*torch.log(1-a+1e-8))
    # print(-y*torch.log(a+1e-8)-(1-y)*torch.log(1-a+1e-8))
    return -torch.sum(y*torch.log(a+1e-8)+(1-y)*torch.log(1-a+1e-8))