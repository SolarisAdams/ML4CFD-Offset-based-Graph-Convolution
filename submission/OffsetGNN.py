import os
import time
import random
import math 
import datetime as dt

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity
from torch.nn import Linear
import airfrans as af
import matplotlib.tri as tri

from lips import get_root_path
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from lips.dataset.airfransDataSet import download_data
from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative

import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from lips.dataset.airfransDataSet import AirfRANSDataSet
import vtkmodules
import vtkmodules.util
import vtkmodules.util.numpy_support
from sklearn.neighbors import NearestNeighbors

submit = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12, help="Number of epochs to train.")
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--hidden1", type=int, default=256)
parser.add_argument("--hidden2", type=int, default=256)
parser.add_argument("--hidden3", type=int, default=128)
parser.add_argument("--lr", type=float, default=1.5e-4)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--w1", type=float, default=0.1)
parser.add_argument("--w2", type=float, default=0.2)
parser.add_argument("--w3", type=float, default=1.0)
parser.add_argument("--w4", type=float, default=2.5e-4)
parser.add_argument("--w5", type=float, default=0.1)
parser.add_argument("--w6", type=float, default=0.2)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--noise1", type=float, default=0.00001)
parser.add_argument("--noise2", type=float, default=0.00000)
parser.add_argument("--noise3", type=float, default=0.00001)
parser.add_argument("--bagging_k", type=int, default=1)
parser.add_argument("--k", type=int, default=1)
parser.add_argument("--lr_decay", type=float, default=0.99995)
parser.add_argument("--gpu", type=int, default=0)


if submit:
    args = parser.parse_args([])
else:
    args = parser.parse_args()


print(args)

class _TensorizedDatasetIter(object):
    def __init__(self, dataset, dataset2, dataset3, batch_size, batchsize2, batchsize3, device):
        self.dataset = dataset
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.batch_size = batch_size
        self.batchsize2 = batchsize2
        self.batchsize3 = batchsize3
        self.index = 0
        self.index2 = 0
        self.index3 = 0
        self.device = device

    def __iter__(self):
        return self
    
    def _next_indices(self):
        num_items = self.dataset.shape[0]
        if self.index >= num_items:
            raise StopIteration
        end_idx = self.index + self.batch_size
        end_idx2 = self.index2 + self.batchsize2
        end_idx3 = self.index3 + self.batchsize3
        if end_idx > num_items:
            end_idx = num_items
        batch = self.dataset[self.index : end_idx]
        batch2 = self.dataset2[self.index2 : end_idx2]
        batch = torch.cat([batch, batch2], dim=0)
        self.index += self.batch_size
        self.index2 += self.batchsize2
        self.index3 += self.batchsize3
        return batch

    def __next__(self):
        batch = self._next_indices()
        return batch.clone().to(self.device)
class CustomBalancedDataset(torch.utils.data.IterableDataset):
    def __init__(self, graph, batch_size, device):
        self.num_nodes = graph.num_nodes()
        self.graph = graph
        self.surface_mask = graph.ndata["features"][:, 8]>0.1
        self.close_mask = graph.ndata["features"][:, 4]<0.05
        self._id_tensor_surface = self.surface_mask.nonzero().repeat(1,8).flatten().to(device)
        self._id_tensor_close = self.close_mask.nonzero().flatten().to(device)
        self._id_tensor = torch.arange(self.num_nodes, dtype=torch.int64, device=device)
        self._indices_surface = torch.arange(self._id_tensor_surface.shape[0], dtype=torch.int64)
        self._indices_close = torch.arange(self._id_tensor_close.shape[0], dtype=torch.int64)
        self._indices = torch.arange(self._id_tensor.shape[0], dtype=torch.int64)
        self.device = device
        num_samples = self.num_nodes
        self.batch_size = batch_size
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        self.batch_size_surface = self._id_tensor_surface.shape[0]// num_batches
        self.batch_size_close = self._id_tensor_close.shape[0]// num_batches
        
    def shuffle(self):
        np.random.shuffle(self._indices.numpy())
        np.random.shuffle(self._indices_surface.numpy())
        np.random.shuffle(self._indices_close.numpy())

    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        num_batches = (self.num_nodes + self.batch_size - 1) // self.batch_size
        self.batch_size_surface = self._id_tensor_surface.shape[0]// num_batches
        self.batch_size_close = self._id_tensor_close.shape[0]// num_batches
        

    def __iter__(self):
        indices  = self._indices
        id_tensor = self._id_tensor[indices]
        id_tensor_surface = self._id_tensor_surface[self._indices_surface]
        id_tensor_close = self._id_tensor_close[self._indices_close]
        return _TensorizedDatasetIter(id_tensor, id_tensor_surface, id_tensor_close, self.batch_size, self.batch_size_surface, self.batch_size_close, self.device)

    def __len__(self):
        num_samples = self.num_nodes
        return (num_samples + self.batch_size - 1) // self.batch_size


class CoordConv(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, coord_dim=2, dist_fn="eculidean", batch_norm=False):
        super(CoordConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = nn.ELU
        self.kernel = nn.Sequential(
            nn.Linear(coord_dim, h_feat),
            self.activation(),
            nn.Linear(h_feat, in_feat),
        )
        self.dist_fn = dist_fn
        self.mlp_self = nn.Sequential(
            nn.Linear(in_feat, h_feat),
            self.activation(),
            nn.Linear(h_feat, h_feat),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_feat+h_feat, h_feat),
            self.activation(),
            nn.Linear(h_feat, h_feat),
            self.activation(),
            nn.Linear(h_feat, h_feat),
            self.activation(),
            nn.Linear(h_feat, out_feat),
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)


    def forward(self, g, feat, offsets):
        with g.local_scope():
            g.srcdata["x"] = feat
            g.dstdata["x_"] = feat[:g.num_dst_nodes()]
            kernel_weights = self.kernel(offsets)
            if self.dist_fn == "eculidean":
                g.edata["w"] = dgl.ops.edge_softmax(g, 1/(torch.norm(offsets, p=1, dim=1)+0.001))
            elif self.dist_fn == "mixed":
                g.edata["w"] = dgl.ops.edge_softmax(g, 1/(torch.norm(offsets, p=1, dim=1)**2+0.00001))
            elif self.dist_fn == "cosine":
                g.edata["w"] = dgl.ops.edge_softmax(g, F.cosine_similarity(offsets, offsets.mean(dim=0), dim=1))
            else:
                raise NotImplementedError
            g.edata["e"] = g.edata["w"].unsqueeze(1) * kernel_weights
            g.update_all(fn.u_mul_e("x", "e", "m"), fn.sum("m", "x"))
            h = self.mlp(torch.cat((g.dstdata["x"], self.mlp_self(g.dstdata["x_"])), dim=1))
            if self.batch_norm:
                h = self.bn(h)
            return h

class CoordGNN(nn.Module):
    def __init__(self, in_feats, h_feats1, h_feats2, h_feats3, h_targets, num_learners):
        super(CoordGNN, self).__init__()
        self.conv1 = CoordConv(in_feats, h_feats1, h_feats1)
        self.conv2 = CoordConv(in_feats, h_feats2, h_feats2)
        self.conv3 = CoordConv(in_feats, h_feats3, h_feats3)
        self.conv4 = CoordConv(h_feats1, h_feats1, h_feats1)
        self.conv5 = CoordConv(h_feats2, h_feats2, h_feats2)
        self.conv6 = CoordConv(h_feats3, h_feats3, h_feats3)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU
        self.skip1 = nn.Linear(in_feats, h_feats1)
        self.skip2 = nn.Linear(in_feats, h_feats2)
        self.skip3 = nn.Linear(in_feats, h_feats3)
        self.skip4 = nn.Linear(in_feats, h_feats1)
        self.skip5 = nn.Linear(in_feats, h_feats2)
        self.skip6 = nn.Linear(in_feats, h_feats3)
        self.act1 = self.activation()
        self.act2 = self.activation()
        self.out1 = nn.Sequential(
            nn.Linear(h_feats1*2, h_feats1),
            self.activation(),
            nn.Linear(h_feats1, h_feats1),
            self.activation(),
            nn.Linear(h_feats1, h_feats1),
            self.activation(),
            nn.Linear(h_feats1, 2),

        )
        self.out2 = nn.Sequential(
            nn.Linear(h_feats2*2, h_feats2),
            self.activation(),   
            nn.Linear(h_feats2, h_feats2),
            self.activation(),
            nn.Linear(h_feats2, h_feats2),
            self.activation(),
            nn.Linear(h_feats2, h_feats2),
            self.activation(),
            nn.Linear(h_feats2, 1)
        )
        self.out3 = nn.Sequential(
            nn.Linear(h_feats3*2, h_feats3),
            self.activation(),
            nn.Linear(h_feats3, h_feats3),
            self.activation(),
            nn.Linear(h_feats3, 1)
        )
    def forward(self, blocks, feat):
        if self.training:
            feat1 = feat + torch.randn_like(feat).to(feat.device).mul(args.noise1)
            feat2 = feat + torch.randn_like(feat).to(feat.device).mul(args.noise2)
            feat3 = feat + torch.randn_like(feat).to(feat.device).mul(args.noise3)
        else:
            feat1 = feat
            feat2 = feat
            feat3 = feat
        h0 = self.skip1(feat1[:blocks[0].num_dst_nodes()])
        h0_ = self.skip2(feat2[:blocks[0].num_dst_nodes()])
        h0__ = self.skip3(feat3[:blocks[0].num_dst_nodes()])

        h2 = self.skip4(feat1[:blocks[1].num_dst_nodes()])
        h2_ = self.skip5(feat2[:blocks[1].num_dst_nodes()])
        h2__ = self.skip6(feat3[:blocks[1].num_dst_nodes()])

        h = self.conv1(blocks[0], feat, offsets=blocks[0].edata['offsets'].to(feat.device))
        h_ = self.conv2(blocks[0], feat, offsets=blocks[0].edata['offsets'].to(feat.device))
        h__ = self.conv3(blocks[0], feat, offsets=blocks[0].edata['offsets'].to(feat.device))
        h = h+h0
        h_ = h_+h0_
        h__ = h__+h0__
        h1 = h[:blocks[1].num_dst_nodes()]
        h1_ = h_[:blocks[1].num_dst_nodes()]
        h1__ = h__[:blocks[1].num_dst_nodes()]
        h = self.act1(h)
        h_ = self.act1(h_)
        h__ = self.act1(h__)
        h = self.conv4(blocks[1], h, offsets=blocks[1].edata['offsets'].to(feat.device))
        h_ = self.conv5(blocks[1], h_, offsets=blocks[1].edata['offsets'].to(feat.device))
        h__ = self.conv6(blocks[1], h__, offsets=blocks[1].edata['offsets'].to(feat.device))
        h = h+h1
        h_ = h_+h1_
        h__ = h__+h1__
        h = torch.cat([h, h2], dim=1)
        h_ = torch.cat([h_, h2_], dim=1)
        h__ = torch.cat([h__, h2__], dim=1)
        h = self.act2(h)
        h_ = self.act2(h_)
        h__ = self.act2(h__)
        out1 = self.out1(h)
        out3 = self.out3(h__)
        out2 = self.out2(h_)
        return torch.cat([out1, out2, out3], dim=1)

class AugmentedSimulator():
    def __init__(self,benchmark,**kwargs):
        self.name = "AirfRANSSubmission"
        self.benchmark = benchmark

        self.hparams = kwargs
        use_cuda = torch.cuda.is_available()
        if submit:
            self.device = 'cuda:0' if use_cuda else 'cpu' 
        else:
            self.device = args.gpu 
        torch.cuda.set_device(self.device)

        if use_cuda:
            print('Using GPU')
        else:
            print('Using CPU')
        self.path_to_simulations = benchmark.benchmark_path
        print("path:", self.path_to_simulations)
        self.processed = {}        
        if submit:
            self.processed["train"] = self.process_dataset(benchmark.train_dataset, training=True)
            self.processed["test"] = self.process_dataset(benchmark._test_dataset, training=False)
            self.processed["test_ood"] = self.process_dataset(benchmark._test_ood_dataset, training=False)
        else:
            if os.path.exists("processed_dataset_211imp.pt"):
                self.processed = torch.load("processed_dataset_211imp.pt")
            else:
                self.processed["train"] = self.process_dataset(benchmark.train_dataset, training=True)
                self.processed["test"] = self.process_dataset(benchmark._test_dataset, training=False)
                self.processed["test_ood"] = self.process_dataset(benchmark._test_ood_dataset, training=False)
                torch.save(self.processed, "processed_dataset_211imp.pt")

    def process_dataset(self, dataset, training: bool):
        name = dataset.name
        if name in self.processed: 
            return self.processed[name]
        simulation_names = dataset.extra_data['simulation_names']
        # simulation_sizes = dataset.get_simulations_sizes()
        p = 0
        graphs = []
        shape_features = []
        feat_names = ["x-position", "y-position", "x-inlet_velocity","y-inlet_velocity","distance_function","x-normals","y-normals"]
        target_names = ["x-velocity","y-velocity","pressure","turbulent_viscosity"]
        normalize_params = {"means":[62, 5, 12.5, 62, 5, 42, 10, -475, 0.0008],
                            "stds":[20, 6, 4.2, 20, 6, 30, 31, 2800, 0.003],
                            "maxs":[0.025, 0.065]}
        
        print("Processing the dataset")
        for i, (name, size) in enumerate(simulation_names):
            size = int(size)
            x_pos = torch.tensor(dataset.data["x-position"][p:p+size]).to(torch.float32)
            y_pos = torch.tensor(dataset.data["y-position"][p:p+size]).to(torch.float32)

            pos = torch.stack([x_pos, y_pos], dim=1)
            path = os.path.join(self.path_to_simulations, name, name+"_internal.vtu")
            reader = vtkmodules.vtkIOXML.vtkXMLUnstructuredGridReader()
            reader.SetFileName(path)
            reader.Update()
            unstructured_grid = reader.GetOutput()
            # 获取单元格数据
            cells = unstructured_grid.GetCells().GetData()
            cells = vtkmodules.util.numpy_support.vtk_to_numpy(cells).reshape(-1, 5)
            cells[:, 0] = cells[: , 4]

            srcs = torch.tensor(cells[:, :4].flatten())
            dsts = torch.tensor(cells[:, 1:].flatten())

            surface = torch.tensor(dataset.extra_data["surface"][p:p+size]).to(bool)
            surface_pos = pos[surface]
            points_middle = surface_pos[(surface_pos[:, 0]<0.3) & (surface_pos[:, 0]>0.2)]
            points_middle_max = torch.max(points_middle[: , 1])
            points_middle_min = torch.min(points_middle[: , 1])
            central = torch.tensor([0.25, (points_middle_max+points_middle_min)/2])
            offsets = surface_pos-central
            angles = torch.arctan2(offsets[: , 1], offsets[: , 0])
            sorted_index = torch.argsort(angles)
            surface_pos = surface_pos[sorted_index]
            surface_id_sorted = surface.nonzero().squeeze()[sorted_index]
            
            surface_srcs = surface_id_sorted
            surface_dsts = torch.cat([surface_id_sorted[1:], surface_id_sorted[:1]], dim=0)

            srcs = torch.cat([srcs, surface_srcs], dim=0)
            dsts = torch.cat([dsts, surface_dsts], dim=0)
            srcs, dsts = torch.cat((srcs, dsts), dim=0), torch.cat((dsts, srcs), dim=0)

            knn_graph = dgl.knn_graph(torch.stack([x_pos, y_pos], dim=1).to(self.device), k=10, algorithm="bruteforce-sharemem").cpu()
            knn_graph.add_edges(srcs, dsts)

            graph = knn_graph
            graph = dgl.to_simple(graph)
            graph = dgl.add_self_loop(graph)
            graph.edata.pop("count")




            shape_feature = []
            simulation_name_splitted = name.split("_")
            shape_feature.append(float((float(simulation_name_splitted[-1]) - normalize_params["means"][0])/ normalize_params["stds"][0]))
            airfoil_params = np.array([float(x) for x in simulation_name_splitted[4:]])
            xlen = 20
            xs = np.arange(0, 1, 1/xlen)
            ys, dys = af.naca_generator.camber_line(airfoil_params[:-1], xs)
            rs = af.naca_generator.thickness_dist(airfoil_params[-1]/100, xs)
            ys = ys / normalize_params["maxs"][0]
            rs = rs / normalize_params["maxs"][1]
            shape_feature.extend(ys)
            shape_feature.extend(rs)
            shape_feature = torch.tensor(shape_feature).to(torch.float32)
            shape_features.append(shape_feature)


            for feat_name in ["distance_function","x-normals","y-normals"]:
                graph.ndata[feat_name] = torch.tensor(dataset.data[feat_name][p:p+size]).to(torch.float32)
            for target_name in ["x-velocity","y-velocity","pressure","turbulent_viscosity"]:
                graph.ndata[target_name] = torch.tensor(dataset.data[target_name][p:p+size]).to(torch.float32)

            features = []
            x_vel = dataset.data["x-inlet_velocity"][p]
            y_vel = dataset.data["y-inlet_velocity"][p]
            vel_dir = torch.tensor([np.arctan2(y_vel, x_vel)]).repeat(graph.num_nodes())
            norm = np.sqrt(x_vel**2+y_vel**2)
            x_vel = x_vel/norm
            y_vel = y_vel/norm
            x_dir = (x_vel, y_vel)
            y_dir = (-y_vel, x_vel)
            offset_x = x_pos - 1
            offset_y = y_pos
            offset_x_ = x_pos
            M = np.array((x_dir, y_dir))
            coord = np.array((offset_x, offset_y)).T
            new_coord = np.dot(coord, M.T)/2
            coord_ = np.stack((offset_x_, offset_y), 0).T
            new_coord_ = np.dot(coord_, M.T)/2
            coord = torch.from_numpy(coord)
            surface = ((graph.ndata["x-normals"]**2+graph.ndata["y-normals"]**2)>0.1)
            surface_coord = coord[surface]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(surface_coord)
            distances, indices = nbrs.kneighbors(coord)
            indices = indices.flatten()
            # distances = distances.flatten()
            surface_normal = torch.stack((graph.ndata["x-normals"][surface], graph.ndata["y-normals"][surface]), dim = 1)
            normal = surface_normal[indices]
            normal[indices==0] = coord[indices==0]
            normal[indices==0] = -1*normal[indices==0] / (normal[indices==0].norm(p=2, dim=1, keepdim=True)+1e-8)
            normal_dir = np.arctan2(normal[:, 1], normal[:, 0])
            offset = coord - surface_coord[indices]
            offset_dir = offset / (offset.norm(p=2, dim=1, keepdim=True)+0.000000001)
            dot_products = torch.sum(offset * normal, dim=1)
            normal_lengths_squared = torch.sum(normal ** 2, dim=1)+1e-8
            projection_factors = dot_products / normal_lengths_squared
            offset = projection_factors.unsqueeze(1)*normal
            rescaled_offset = (projection_factors.sign() * projection_factors.abs()**0.2).unsqueeze(1)*offset
            distances = offset.norm(p=2, dim=1).flatten()


            for feat_name in feat_names:
                features.append(torch.tensor(dataset.data[feat_name][p:p+size]))
            features[2] = (features[2]-normalize_params["means"][3])/normalize_params["stds"][3]
            features[3] = (features[3]-normalize_params["means"][4])/normalize_params["stds"][4]
            features.append(features[4]**0.2)
            features.append(surface.to(torch.float32))
            features.append(new_coord[:, 0])
            features.append(new_coord[:, 1])
            features.append(new_coord_[:, 0])
            features.append(new_coord_[:, 1])
            features.append(normal[:, 0])
            features.append(normal[:, 1])
            features.append(offset[:, 0])
            features.append(offset[:, 1])
            features.append(rescaled_offset[:, 0])
            features.append(rescaled_offset[:, 1])
            features.append(distances)
            features.append(distances**0.2)
            features.append(indices/max(indices+1))
            features.append(vel_dir)
            features.append(normal_dir)
            features.append(vel_dir-normal_dir)
            features = np.stack(features, 1)
            # print(graph, shape_feature.unsqueeze(0).repeat(graph.num_nodes(), 1).shape)


            graph.ndata["features"] = torch.tensor(features).to(torch.float32)
            graph.ndata["sim_ids"] = torch.zeros(graph.number_of_nodes(), dtype=torch.long).fill_(i)
            features = []
            for feat_name in target_names:
                features.append(torch.tensor(dataset.data[feat_name][p:p+size]))
            for j in range(4):
                features[j] = (features[j]-normalize_params["means"][j+5])/normalize_params["stds"][j+5]
            features = np.stack(features, 1)
            graph.ndata["targets"] = torch.from_numpy(features).to(torch.float32)
            srcs, dsts = graph.edges()
            # distance
            x_offset = torch.tensor(dataset.data["x-position"][p:p+size][dsts] - dataset.data["x-position"][p:p+size][srcs]).to(torch.float32)
            y_offset = torch.tensor(dataset.data["y-position"][p:p+size][dsts] - dataset.data["y-position"][p:p+size][srcs]).to(torch.float32)
            graph.edata["offsets"] = torch.stack([x_offset, y_offset], 1).to(torch.float32)
            # graph.ndata["means"] = graph.ndata["targets"].mean(0).unsqueeze(0).repeat(graph.num_nodes(), 1)
            # graph.ndata["stds"] = graph.ndata["targets"].std(0).unsqueeze(0).repeat(graph.num_nodes(), 1)

            p += size
            graphs.append(graph)
            if (i+1)%10==0:
                print(f"Preprocessing {i+1} of {len(simulation_names)} simulation...", graph)

        merged_graphs = dgl.batch(graphs).long()
        shape_features = torch.stack(shape_features, 0)
        means = []
        stds = []
        if training:
            for i, graph in tqdm(enumerate(graphs)):
                means.append(graph.ndata["targets"].mean(0))
                stds.append(graph.ndata["targets"].std(0))
            means = torch.stack(means, 0)
            stds = torch.stack(stds, 0)
        else:
            for i, graph in tqdm(enumerate(graphs)):
                means.append(graph.ndata["targets"])
                stds.append(graph.ndata["targets"])
            means = torch.cat(means, 0)
            stds = torch.cat(stds, 0)
            means = means.mean(0, keepdim=True).repeat(len(graphs), 1)
            stds = stds.std(0, keepdim=True).repeat(len(graphs), 1)


        return merged_graphs, shape_features, means, stds

    def train(self,train_dataset, save_path=None):
        graphs, shape_features, means, stds = self.process_dataset(dataset=train_dataset,training=True)
        print("Start training")
        if submit:
            self.model = global_train(self.device, graphs, shape_features, means, stds)
        else:
            self.model = global_train(self.device, graphs, shape_features, means, stds, \
                                      self.processed["test"], self.processed["test_ood"])
        print("Training done")

    def predict(self,dataset,**kwargs):
        normalize_params = {"means":[62, 4.5, 12.5, 62, 5, 42, 10, -475, 0.0008],
                            "stds":[20, 5.5, 4.2, 20, 6.5, 30, 31, 2800, 0.003],
                            "maxs":[0.025, 0.065]}
        print(dataset)
        graphs, shape_features, means, stds = self.process_dataset(dataset=dataset,training=False)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 15])
        dataloader = dgl.dataloading.DataLoader(graphs, torch.arange(graphs.num_nodes()), sampler,\
                            batch_size=8192, shuffle=False, drop_last=False, num_workers=0, use_uva=True)
        self.model.eval()
        device = self.device
        self.model = self.model.to(device)
        shape_features = shape_features.to(device)
        means, stds = means.to(device), stds.to(device)
        with torch.no_grad():
            test_output = []
            for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                blocks = [block.to(device) for block in blocks]
                x = blocks[0].srcdata["features"].to(device).to(torch.float32)
                x_shape = shape_features[blocks[0].srcdata["sim_ids"].to(device)]
                x = torch.cat([x, x_shape], dim=1)
                y_hat = self.model(blocks, x)
                test_output.append(y_hat.cpu().detach())
                if it % 1000 == 0:
                    print("Iteration {:05d}/{:05d} | Test Set".format(it, len(dataloader)))
        predictions = np.concatenate([i.numpy() for i in test_output])
        predictions = dataset.reconstruct_output(predictions)
        for i, name in enumerate(["x-velocity","y-velocity","pressure","turbulent_viscosity"]):
            std = float(normalize_params["stds"][i+5])
            mean = float(normalize_params["means"][i+5])
            predictions[name] = predictions[name]*std + mean
        return predictions

def slope(x, low, high, square, train, beta=args.beta):
    # lower = low - (high - low)*0.1
    if square:
        x = x**2
    else:
        x = x.abs()
    normalized = (x)/(low)
    # normalized = (x-low)/(high-low)
    if train and beta>0:
        # zero_point = -low/(high-low)
        zero_point=0
        return ((normalized+torch.sqrt(normalized**2+beta))/2) - ((zero_point+np.sqrt(zero_point**2+beta))/2)
    else:
        return normalized
def LossFn(y_pred, y_true, surface_mask, sim_ids, means, stds, square=True, train=False):
    y_true = (y_true-means[sim_ids]) / stds[sim_ids]
    y_pred = (y_pred-means[sim_ids]) / stds[sim_ids]
    y_surface = y_true[surface_mask, :3]
    y_hat_surface = y_pred[surface_mask, :3]
    if train:
        loss0 = torch.mean(slope(y_true[:, 0] - y_pred[:, 0], 0.01, 0.02, square, train))
        loss1 = torch.mean(slope(y_true[:, 1] - y_pred[:, 1], 0.01, 0.02, square, train))
        loss2 = torch.mean(slope(y_true[:, 2] - y_pred[:, 2], 0.0005, 0.001, square, train))
        loss3 = torch.mean(slope(y_true[:, 3] - y_pred[:, 3], 0.05, 0.1, square, train))
        loss4 = torch.mean(slope(y_surface[:, 2] - y_hat_surface[:, 2], 0.002, 0.004, square, train))
        loss5 = torch.mean(slope((y_surface[:, :2] - y_hat_surface[:, :2]).abs().mean(1), 0.001, 0.002, square, train))
    else:
        loss0 = torch.mean(torch.square(y_true[:, 0] - y_pred[:, 0]))
        loss1 = torch.mean(torch.square(y_true[:, 1] - y_pred[:, 1]))
        loss2 = torch.mean(torch.square(y_true[:, 2] - y_pred[:, 2]))
        loss3 = torch.mean(torch.square(y_true[:, 3] - y_pred[:, 3]))
        loss4 = torch.mean(torch.square(y_surface[:, 2] - y_hat_surface[:, 2]))
        loss5 = torch.mean(torch.square((y_surface[:, :2] - y_hat_surface[:, :2]).abs().mean(1)))
    if y_surface.shape[0]<1:
        loss4 = torch.tensor(0)
        loss5 = torch.tensor(0)
    if train:
        if loss3.item()<0.3:
            w4 = 0
        else:
            w4 = args.w4
        return args.w1*loss0 + args.w2*loss1 + args.w3*loss2 + w4*loss3 + args.w5*loss4 + args.w6*loss5, [loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()]
    else:
        return loss0 + loss1 + loss2 + loss3 + loss4, [loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()]

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, a, b):
        outputs = []
        for model in self.models:
            outputs.append(model(a, b))
        return torch.stack(outputs, dim=0).mean(0)
    
def global_train(device, graphs, shape_features, means, stds, test_dataset=None, test_ood_dataset=None):
    models = []
    np.set_printoptions(suppress=True)
    graphs = graphs.to(device)
    for name in graphs.ndata:
        graphs.ndata[name] = graphs.ndata[name].to(device)
    for name in graphs.edata:
        graphs.edata[name] = graphs.edata[name].to(device)
    dataset = CustomBalancedDataset(graphs, args.batch_size, graphs.device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 15])
    train_dataloader = dgl.dataloading.DataLoader(graphs, dataset, sampler,\
                                        batch_size=256, shuffle=True, device=device, drop_last=False, num_workers=0, use_uva=False)
 
    shape_features = shape_features.to(device)
    means, stds = means.to(device), stds.to(device)

    if test_dataset:
        test_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        test_graphs, test_shape_feats, test_means, test_stds = test_dataset
        test_dataloader = dgl.dataloading.DataLoader(test_graphs, torch.arange(test_graphs.num_nodes()), test_sampler,\
                            batch_size=4096, shuffle=True, drop_last=False, num_workers=0, use_uva=True)
        test_shape_feats = test_shape_feats.to(device)
        test_means, test_stds = test_means.to(device), test_stds.to(device)

    if test_ood_dataset:
        test_ood_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        test_ood_graphs, test_ood_shape_feats, test_ood_means, test_ood_stds = test_ood_dataset
        test_ood_dataloader = dgl.dataloading.DataLoader(test_ood_graphs, torch.arange(test_ood_graphs.num_nodes()), test_ood_sampler,\
                            batch_size=4096, shuffle=True, drop_last=False, num_workers=0, use_uva=True)
        test_ood_shape_feats = test_ood_shape_feats.to(device)
        test_ood_means, test_ood_stds = test_ood_means.to(device), test_ood_stds.to(device)

    for bagging_i in range(args.bagging_k):
        print("Training bagging:", bagging_i)
        inner_models = [CoordGNN(66, args.hidden1, args.hidden2, args.hidden3, 4, args.k).to(device) for _ in range(args.k)]
        optimizers = [torch.optim.Adam(inner_models[_].parameters(), lr=args.lr, weight_decay=args.weight_decay) for _ in range(args.k)]
        lr_schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay) for optimizer in optimizers]
        print("Model got. Start training...")
        running_mean_loss = 1000
        running_mean_losses = [1000, 1000, 1000, 1000, 1000, 1000]

        for epoch in range(args.epochs):
            inner_models = [model.train() for model in inner_models]
            total_loss = 0
            print("Epoch:", epoch, "starts.")
            tick = time.time()
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                if epoch==0 and it==0:
                    print(blocks)
                blocks = [block.to(device) for block in blocks]
                x = blocks[0].srcdata["features"].to(device).to(torch.float32)
                x_shape = torch.index_select(shape_features, 0, blocks[0].srcdata["sim_ids"].to(device))
                x = torch.cat([x, x_shape], dim=1)
                y = blocks[1].dstdata["targets"].to(device).to(torch.float32)
                surface_mask = blocks[1].dstdata["features"][:, 8]>0.5
                for inner_bagging_i in range(args.k):
                    model = inner_models[inner_bagging_i]
                    optimizer = optimizers[inner_bagging_i]
                    lr_scheduler = lr_schedulers[inner_bagging_i]
                    y_hat = model(blocks, x)
                    loss, losses = LossFn(y_hat, y, surface_mask, blocks[1].dstdata["sim_ids"], means, stds, square=epoch>-1, train=True)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    if running_mean_loss>500:
                        running_mean_loss = loss.item()
                        running_mean_losses = [t for t in losses]
                    else:
                        running_mean_loss = loss.item()*0.001 + running_mean_loss * 0.999
                        running_mean_losses = [t*0.001 + r*0.999 for t,r in zip(losses, running_mean_losses)]
                    if epoch>=6:
                        lr_scheduler.step()
                if it > args.steps:
                    break
                if it % 500 == 0:
                    print("    Iteration {:05d}/{:05d} | Train Loss: {:.4f} \t| Detailed losses:{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, {:.3f}".format(it, len(train_dataloader), running_mean_loss, *running_mean_losses))

            tock = time.time()
            print("Epoch {:05d} | Train Loss:{:.4f} Time:{:.2f}".format(epoch, total_loss / (it + 1), tock-tick))
            with torch.no_grad():
                inner_models = [model.eval() for model in inner_models]
                if test_dataset:
                    total_loss = 0
                    # running_mean_loss = 1000
                    losses_list = []
                    for it, (input_nodes, output_nodes, blocks) in enumerate(test_dataloader):
                        blocks = [block.to(device) for block in blocks]
                        x = blocks[0].srcdata["features"].to(device).to(torch.float32)
                        x_shape = test_shape_feats[blocks[0].srcdata["sim_ids"].to(device)]
                        x = torch.cat([x, x_shape], dim=1)
                        y = blocks[-1].dstdata["targets"].to(device).to(torch.float32)
                        surface_mask = blocks[1].dstdata["features"][:, 8]>0.5
                        y_hats = []
                        for inner_bagging_i in range(args.k):
                            model = inner_models[inner_bagging_i]
                            y_hats.append(model(blocks, x))
                        y_hat = torch.stack(y_hats, dim=0).mean(dim=0)
                        loss, losses = LossFn(y_hat, y, surface_mask, blocks[-1].dstdata["sim_ids"], test_means, test_stds)
                        losses_list.append(losses)
                        total_loss += loss.item() 
                        if it>300:
                            break
                    print("  Test set loss:", total_loss/(it+1), np.array(losses_list).mean(axis=0))
                if test_ood_dataset:
                    total_loss = 0
                    losses_list = []
                    # running_mean_loss = 1000
                    for it, (input_nodes, output_nodes, blocks) in enumerate(test_ood_dataloader):
                        blocks = [block.to(device) for block in blocks]
                        x = blocks[0].srcdata["features"].to(device).to(torch.float32)
                        x_shape = test_ood_shape_feats[blocks[0].srcdata["sim_ids"].to(device)]
                        x = torch.cat([x, x_shape], dim=1)
                        y = blocks[-1].dstdata["targets"].to(device).to(torch.float32)
                        surface_mask = blocks[1].dstdata["features"][:, 8]>0.5
                        y_hats = []
                        for inner_bagging_i in range(args.k):
                            model = inner_models[inner_bagging_i]
                            y_hats.append(model(blocks, x))
                        y_hat = torch.stack(y_hats, dim=0).mean(dim=0)
                        loss, losses = LossFn(y_hat, y, surface_mask, blocks[-1].dstdata["sim_ids"], test_ood_means, test_ood_stds)
                        losses_list.append(losses)
                        total_loss += loss.item() 
                        if it>300:
                            break
                    print("  Test OOD set loss:", total_loss/(it+1), np.array(losses_list).mean(axis=0))
                    print()
                inner_models = [model.train() for model in inner_models]
        inner_models = [model.to("cpu").eval() for model in inner_models]
        models.extend(inner_models)
    model = EnsembleModel(models)
    return model
