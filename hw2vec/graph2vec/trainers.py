import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

from hw2vec.graph2vec.models import *
from hw2vec.utilities import *

class BaseTrainer:
    def __init__(self, cfg):
        self.config = cfg
        self.min_test_loss = np.Inf
        self.task = None
        self.metrics = {}
        self.model = None
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def build(self, model, path=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=5e-4)

    def visualize_embeddings(self, data_loader, path=None):
        save_path = "./visualize_embeddings/" if path is None else Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        embeddings, hw_names = self.get_embeddings(data_loader)

        with open(str(save_path / "vectors.tsv"), "w") as vectors_file, \
             open(str(save_path / "metadata.tsv"), "w") as metadata_file:

            for embed, name in zip(embeddings, hw_names):
                vectors_file.write("\t".join([str(x) for x in embed.detach().cpu().numpy()[0]]) + "\n")
                metadata_file.write(name+"\n")

    def get_embeddings(self, data_loader):
        embeds = []
        hw_names = []

        with torch.no_grad():
            self.model.eval()

            for data in data_loader:
                data.to(self.config.device)
                embed_x, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
                embeds.append(embed_x)
                hw_names += data.hw_name

        return embeds, hw_names

    def metric_calc(self, loss, labels, preds, header):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        conf_mtx = str(confusion_matrix(labels, preds)).replace('\n', ',')
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")

        self.metric_print(loss, acc, f1, conf_mtx, precision, recall, header)

        if header == "Test" and (self.min_test_loss >= loss):
            self.min_test_loss = loss
            self.metrics["acc"] = acc
            self.metrics["f1"] = f1
            self.metrics["conf_mtx"] = conf_mtx
            self.metrics["precision"] = precision
            self.metrics["recall"] = recall

    def metric_print(self, loss, acc, f1, conf_mtx, precision, recall, header):
        print("%s Loss: %4f" % (header, loss) +
            ", %s Accuracy: %.4f" % (header, acc) +
            ", %s F1 score: %.4f" % (header, f1) +
            ", %s Confusion_matrix: %s" % (header, conf_mtx) +
            ", %s Precision: %.4f" % (header, precision) +
            ", %s Recall: %.4f" % (header, recall))

class GraphTrainer(BaseTrainer):
    ''' trainer for graph classification ''' 
    def __init__(self, cfg, class_weights=None):
        super().__init__(cfg)
        self.task = "TJ"
        if class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:    
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights.float().to(cfg.device))

    def train(self, data_loader, valid_data_loader):
        tqdm_bar = tqdm(range(self.config.epochs))

        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0

            for data in data_loader:
                self.optimizer.zero_grad()
                data.to(self.config.device)

                loss_train = self.train_epoch_tj(data)
                loss_train.backward()
                self.optimizer.step()
                acc_loss_train += loss_train.detach().cpu().numpy()

            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))

            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx, data_loader, valid_data_loader)

    # @profileit
    def train_epoch_tj(self, data):
        output, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
        output = self.model.mlp(output)
        output = F.log_softmax(output, dim=1)

        loss_train = self.loss_func(output, data.label)
        return loss_train

    # @profileit
    def inference_epoch_tj(self, data):
        output, attn = self.model.embed_graph(data.x, data.edge_index, data.batch)
        output = self.model.mlp(output)
        output = F.log_softmax(output, dim=1)

        loss = self.loss_func(output, data.label)
        return loss, output, attn
                
    def inference(self, data_loader):
        labels = []
        outputs = []
        node_attns = []
        total_loss = 0
        folder_names = []
        
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(data_loader):
                data.to(self.config.device)

                loss, output, attn = self.inference_epoch_tj(data)
                total_loss += loss.detach().cpu().numpy()

                outputs.append(output.cpu())
                
                if 'pool_score' in attn:
                    node_attn = {}
                    node_attn["original_batch"] = data.batch.detach().cpu().numpy().tolist()
                    node_attn["pool_perm"] = attn['pool_perm'].detach().cpu().numpy().tolist()
                    node_attn["pool_batch"] = attn['batch'].detach().cpu().numpy().tolist()
                    node_attn["pool_score"] = attn['pool_score'].detach().cpu().numpy().tolist()
                    node_attns.append(node_attn)

                labels += np.split(data.label.cpu().numpy(), len(data.label.cpu().numpy()))

            outputs = torch.cat(outputs).reshape(-1,2).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = torch.LongTensor(labels).detach()
            outputs_tensor = torch.FloatTensor(outputs).detach()
            preds = outputs_tensor.max(1)[1].type_as(labels_tensor).detach()

        return avg_loss, labels_tensor, outputs_tensor, preds, node_attns

    def evaluate(self, epoch_idx, data_loader, valid_data_loader):
        train_loss, train_labels, _, train_preds, train_node_attns = self.inference(data_loader)
        test_loss, test_labels, _, test_preds, test_node_attns = self.inference(valid_data_loader)

        print("")
        print("Mini Test for Epochs %d:"%epoch_idx)

        self.metric_calc(train_loss, train_labels, train_preds, header="Train")
        self.metric_calc(test_loss,  test_labels,  test_preds,  header="Test")

        if self.min_test_loss >= test_loss:
            self.model.save_model(str(self.config.model_path_obj/"model.cfg"), str(self.config.model_path_obj/"model.pth"))

            #TODO: store the attn_weights right here. 

        # on final evaluate call
        if(epoch_idx==self.config.epochs):
            self.metric_print(self.min_test_loss, **self.metrics, header="Best")


class Evaluator(BaseTrainer):
    def __init__(self, cfg, task):
        super().__init__(cfg)
        self.task = task