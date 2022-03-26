import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import torch

# class GradReverse(Function):
#     def __init__(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         return (grad_output * -self.lambd)
# def grad_reverse(x, lambd=1.0):
#     return GradReverse(lambd)(x)

class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        feat_dim = args.in_features
        self.fc1_1 = nn.Linear(feat_dim, feat_dim)
        self.fc2_1 = nn.Linear(feat_dim, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        # x_out = F.sigmoid(x)
        x_out = x.squeeze(-1)
        return x_out


class Classifier(nn.Module):
    def __init__(self, args, nclass=None):
        super(Classifier, self).__init__()
        self.args = args
        feat_dim = args.in_features
        num_class = args.num_class if nclass is None else nclass
        self.num_class = num_class
        if args.CE_loss is not 'angular':
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, int(feat_dim / 2)),
                nn.BatchNorm1d(int(feat_dim / 2)),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(args.dropout)
            )
            self.fc = nn.Linear(int(feat_dim/2), num_class)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim)
            )
            self.weight = nn.Linear(feat_dim, num_class, bias=False)


    def forward(self, inputs):
        batch_size, n_domain, num_feat = inputs.size()
        inputs = inputs.view(batch_size * n_domain, num_feat)
        inputs = F.normalize(inputs, p=2, dim=-1)
        outputs = self.classifier(inputs)
        weight_norm = []
        if self.args.CE_loss == 'angular':
            for i in range(self.num_class):
                weight_norm.append(F.normalize(self.weight.weight[i], p=2, dim=-1))
            outputs = F.normalize(outputs, p=2, dim=-1)
            outputs = F.linear(outputs, torch.stack(weight_norm, dim=0))
        else:
            outputs = self.fc(outputs)
        outputs = outputs.view(batch_size, n_domain, -1)
        return outputs

class Domain_Classifier(nn.Module):
    def __init__(self, args):
        super(Domain_Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(args.in_features, args.num_domains)
        )

    def forward(self, inputs):
        return self.classifier(inputs)