# TJ-GNN

## Acknowledgement
I would like to express my deepest appreciation to Dr. Suchismita Roy, my project guide, for
her exceptional guidance and support during the development of this project. Dr. Roy's
expertise, insightful feedback, and encouragement have been instrumental in shaping the
project's direction and outcomes.

I am grateful for the opportunity to work under Dr. Roy's mentorship, and I truly value the
knowledge and skills I have gained throughout this journey. Her commitment to excellence
and passion for the subject matter have inspired and enriched this project.

## Problem Statement:
### Title: Graph Neural Networks for Hardware Trojan Detection
Introduction: Hardware Trojan detection is a critical aspect of ensuring the security and
integrity of integrated circuits. Traditional methods face challenges in detecting subtle and
sophisticated Trojans. This project aims to employ Graph Neural Networks (GNNs) for
hardware Trojan detection, utilizing graph representations of circuit designs to enhance
accuracy and robustness.

### Objectives:
1. Graph Representation: Develop effective graph representations of hardware circuit
designs suitable for GNN-based analysis.
2. GNN Architecture: Design and implement specialized GNN architectures capable of
identifying hardware Trojans.
3. Pooling Strategies: Investigate and apply graph pooling strategies to capture crucial
features and patterns indicative of Trojan presence.
4. Evaluation Metrics: Define and employ appropriate evaluation metrics, including
accuracy, F1 score, precision, recall, and confusion matrix, for assessing Trojan
detection performance.
This project contributes to advancing hardware security by leveraging GNNs to improve
Trojan detection capabilities. The developed models have the potential to enhance current
detection methods, particularly in scenarios where traditional approaches may fall short.

## SOLUTION:
The solution to this problem can be represented broadly, as the following steps:
1.Input Hardware Code for Generation of Data Flow Graph: In this step, the goal is to
represent the hardware description in the form of a graph. The hardware code is
parsed using LALR parser and processed to identify components, features, or entities
within the design. Each of these components, represented as segments of Verilog
code, becomes a node in the graph, and the relationships or connections between
them are represented as edges. For example, if the hardware code describes
interconnected modules or gates, nodes in the graph may correspond to these
modules, and edges represent the connections between them.
2. Data Flow Graph to Vector Generation: Graph Convolutional Neural Network:
Graph convolutional layers: These are applied to the graph representation. These
layers learn node embeddings by aggregating information from neighbouring nodes.
The result is a refined node representation that captures local graph structure.
Pooling Layers: Pooling layers are used to down-sample the graph, reducing its size
while retaining essential information. Pooling operations may involve selecting
important nodes or consolidating information from groups of nodes.
Readout Layers: Readout layers aggregate the node embeddings to generate a graph-
level representation. This step is crucial as it captures the overall characteristics of
the entire graph. Different readout methods (e.g., max pooling, mean pooling) may
be used.
Multi-Layer Perceptron (MLP): The final step involves passing the graph-level
representation through an MLP. This layer performs non-linear transformations,
enabling the model to capture complex relationships in the data. The output of the
MLP is the fixed-size vector representation of the input graph.
3.Training the Model: The Graph2Vec model is trained on a dataset of labelled graphs,
where each graph has a corresponding label indicating whether it is Trojan-infected
or Trojan-free. The training process involves adjusting the model's parameters to
minimize a certain loss function, involving a combination of the predicted vector and
the ground truth label.
4.Detecting the Presence of Hardware Trojans: After training, the model can be used
to predict whether a given hardware description contains a Trojan or not. This is
done by inputting the graph representation of the hardware into the trained
Graph2Vec model, obtaining the corresponding vector representation, loading the
previous generated model weights and then using this trained model for
classification.
The dataset used for this has been taken from TRUST-HUB.

## Input Hardware Code to Graph Representation:
In this step, we convert a given input hardware code to a graphical representation. It is
achieved in three stages:
1) Preprocessing
2) Graph Generation
3) Postprocessing
Preprocessing:
Preprocessing includes the following functions:

Flattening: The primary purpose of this function is to take a directory containing Verilog
files, flatten them into a single Verilog file, and storing the flattened content. If a file named
"topModule.v" is already present in the directory, the function returns without further
processing. This avoids unnecessary flattening, if it has already been done. The flatten
method uses the ‘glob’ function from the ‘glob’ module in Python to find all files containing
a specific input_path. For each file that is found, we append its contents to a string, called
flatten_content, which is initially empty. Once we traverse through all the files, we write
the flatten_content string to an output file, called outfile. Ultimately, this process ends up
taking all the contents of files with a specific input_path and putting them into a single
file.

Removing Comments and Underscores: to remove single-line comments (denoted by //)
and remove underscores from a Verilog file. The functions reads the file, processes each
line, and writes the modified content back to the same file. It is a common preprocessing
step in code analysis to simplify subsequent parsing or processing steps, to ensure that
comments or underscores do not interfere with graph generation or other analyses.

Rename the top module: The primary purpose of this function is to rename the top
module in a Verilog file to "top." It achieves this by identifying the module declarations
and determining the module with a single occurrence, assuming it is the top module.
Itinitializes an empty dictionary, called modules_dic, which is meant to store module
names as keys and the module name frequency as the values.

## Graph Generation:
After preprocessing, the hardware code must be converted to graph representation, as a
DFG(Data Flow Graph) or an AST(Abstract Syntax Tree).
Data Flow Graph represents the flow of data through a digital design. It is a graphical
representation that illustrates how data moves between different components or elements in
a digital circuit. Nodes represent operations or computations, Edges represent the flow of data
between nodes and the direction of the edges indicates the direction in which data flows. In
order to produce a DFG, DFGGenerator.process creates an instance of the
VerilogDataflowAnalyzer
 class,
 called
 dataflow_analyzer.
 The
VerilogDataflowAnalyzer.generate outputs a parse tree also with the help of YACC. An instance
of the VerilogGraphGenerator is then created and assigned to the variable name
dfg_graph_generator. The dfg_graph_generator has a binddict, which is a dictionary whose
keys are nodes (the signals) and the values are each node’s associated dataflow object. For
each signal in the dfg_graph_generator binddict, we call the DFGGenerator.generate to create
a signal DFG. Lastly, we merge all the signal DFGs together. The resulting graph is a DFG that
is in JSON format.

## PostProcessing:
The graph obtained G, is in JSON format, which is converted it a NetworkX object, since,
libraries like PyTorch Geometric take a NetworkX graph object as their primary data structure
in their pipelines, so a conversion from JSON format to a NetworkX format is necessary.
For both the DFG and AST, we initialize the NetworkX graph object, which we name nx_graph,
as a NetworkX DiGraph object, which holds directed edges.

## Graph to Vector Representation:
The next step is to normalize the NetworkX graph object by iterating through the nodes of the
nx_graph and giving each node a label that indicates its type.
For the DFG, the type of the node can be numeric, output, input, or signal. The AST node can
have a type of names or pure numeric, otherwise the type remains unchanged. The label type
is then used to convert each one of the nodes into a vectorized representation. This is done
by the normalize function in the DataProcessor class.
It converts the graph g into the matrices X, which represents the node embeddings, and A,
which represents the adjacency information of the graph. The from_networkx has an empty
dictionary, data, which is filled with the graph’s nodes and edges. By calling torch.tensor on
each item in data, we create feature vectors for each key in the dictionary. These feature
vectors represent the node embeddings, X. from_networkx also creates an adjacency matrix
(this represents A) of the NetworkX graph object’s edges by calling torch.LongTensor on the
list of the graph’s edges. This adjacency matrix, called edge_index, becomes an attribute of a
dictionary, called data. Finally, data is converted into a PyTorch geometric Data instance by
calling torch_geometric.data.Data.from_dict on the dictionary and this object is returned by
the function.

## The Graph Convolution:
1) Graph Convolution Layers
2) Pooling Layers
3) Readout Layers
Graph Convolution Layer
The core idea of GCNs is based on message passing between nodes. Each node aggregates
information from its neighbours, transforming its own feature representation. The
aggregation involves weighted combinations of features from neighbouring nodes, where
weights are determined by the edges in the graph. Thus, message passing involves two sub-
phases: Aggregate and Combine. The AGGREGATE function updates the node embeddings
after each k-th iteration to produce X(k) using each node representation hv(k-1) in X(k-1). The
function essentially accumulates the features of the neighboring nodes and produces an
aggregated feature vector av(k) for each layer k. The COMBINE function combines the previous
node feature hv (k-1) with av(k) to output the next feature vector hv(k). The final node embedding
after the message propagation is denoted as Xprop.
The __init__ method initializes the layer with the specified type, input channels, and output
channels. The forward method takes node features (x) and edge indices (edge_index) as input
and applies the GCN operation.
Pooling Layer
The GRAPH_POOL class is designed for graph pooling in a Graph Neural Network (GNN). Graph
pooling is a technique used to down-sample or reduce the size of a graph while retaining
important structural information. In this case, two types of pooling methods are
implemented: SAGPooling (Self-Attention Graph Pooling) and TopKPooling.
The parameters it requires are: type- Specifies the pooling method, either "sagpool" or
"topkpool", in_channels- Number of input channels, typically representing the node features
and poolratio- A parameter specific to the pooling method, e.g., the ratio of nodes to be
retained after pooling, to initialize and then create an instance of either SAGPooling or
TopKPooling based on the specified type. In case of the forward function, takes in parameters:
x- Input node features, edge_index- Graph connectivity information and batch- A vector
indicating which nodes in the batch are part of the same graph. This method passes the input
data through the selected pooling layer (SAGPooling or TopKPooling). The pooling operation
is applied to the input data, and the down-sampled output is returned. The batch parameter
is used to identify the nodes belonging to the same graph during the pooling operation.
Readout Layer
The GRAPH_READOUT class is responsible for performing graph readout in the context of a
graph neural network (GNN). Graph readout involves aggregating information from individual
nodes in a graph to produce a graph-level representation. The specific type of aggregation is
determined by the type parameter, which can be one of three options: "max," "mean," or
"add."
The constructor initializes the GRAPH_READOUT object with a specified type. The type
parameter determines the type of aggregation operation to be used during graph readout.
The forward method defines the forward pass of the module, which describes how input data
is transformed to produce the output. In this case, the input x represents node features, and
batch is a vector that indicates which nodes in the batch belong to the same graph. For each
type of readout operation, a corresponding global pooling function is called on the node
features x. The choice of readout type determines how information is aggregated across
nodes. If type is "max," global_max_pool is applied, which computes the maximum value
along each feature dimension across nodes. If type is "mean," global_mean_pool is applied,
which computes the mean value along each feature dimension across nodes. If type is "add,"
global_add_pool is applied, which sums the node features along each feature dimension
across nodes. The resulting output represents a graph-level representation that captures the
essential information from the individual nodes.

## Training the Model and Evaluation:
In this section we aim to train the dataset, validate the dataset and a set of hyperparameter
configurations to train the GNN model. This is achieved with the help of following classes:
BaseTrainer:
The BaseTrainer class serves as a generic foundation for training GNN models, providing
common functionalities such as model building, visualization of embeddings, metric
calculation, and printing. Subclasses, like GraphTrainer and Evaluator, build upon this base
for specific tasks. BaseTrainer creates an instance of an Adam optimizer, which implements
the Adam algorithm.

Consider each of the following methods:
Initialization:
• Attributes:
• self.config: Stores the configuration object, which holds various hyperparameters
and settings for the training process.
• self.min_test_loss: Keeps track of the minimum test loss during training.
• self.task: Represents the task being performed (initialized as None).
• self.metrics: A dictionary to store various metrics during training.
• self.model: Holds the GNN model being trained.
• Random Seed:
• Sets the random seed for NumPy and PyTorch to ensure reproducibility.
Methods:
1.2.3.4.build(model, path=None):
• Inputs:
• model: The GNN model to be trained.
• path (optional): Path to load a pre-existing model.
• Functionality:
• Initializes the model and an Adam optimizer.
• Optionally loads a pre-existing model from the specified path.
visualize_embeddings(data_loader, path=None):
• Inputs:
• data_loader: DataLoader containing graph data.
• path (optional): Path to save the visualization files.
• Functionality:
• Obtains graph embeddings and hardware names using get_embeddings.
• Saves embeddings in TSV format for visualization.
get_embeddings(data_loader):
• Input:
• data_loader: DataLoader containing graph data.
• Functionality:
• Retrieves graph embeddings and hardware names from the model.
• Returns:
• Tuple containing embeddings and hardware names.
metric_calc(loss, labels, preds, header):
• Inputs:
• loss: Current loss value.
• labels: True labels.

5.•
 preds: Predicted labels.
• header: Indicates whether it's a training or testing metric.
• Functionality:
• Calculates and prints various classification metrics.
• Updates self.metrics and self.min_test_loss if applicable.
metric_print(loss, acc, f1, conf_mtx, precision, recall, header):
• Inputs:
• Metrics to print.
• Functionality:
• Prints the calculated metrics.
GraphTrainer:
The GraphTrainer class specializes in training GNNs for graph classification tasks, specifically
designed for the task labeled "TJ". It leverages the BaseTrainer functionalities and extends
them for the specific requirements of the graph classification task. The training process
involves forward passes, loss computation, and periodic evaluation of the model's
performance.
It inherits from BaseTrainer, meaning it inherits all the attributes and methods from the
BaseTrainer class.

Initialization:
• Attributes:
• self.task: Initialized as "TJ," representing the task specific to this trainer.
• self.loss_func: Defines the loss function for the task (cross-entropy loss).
• Initialization based on Class Weights:
• Checks if the number of classes is less than 2 to determine whether to use class
weights in the loss function.
Methods:
1.2.3.train(data_loader, valid_data_loader):
• Inputs:
• data_loader: DataLoader for training data.
• valid_data_loader: DataLoader for validation data.
• Functionality:
• Trains the GNN model for a specified number of epochs.
• Uses cross-entropy loss and Adam optimizer.
• Periodically evaluates the model using evaluate during training.
train_epoch_tj(data):
• Input:
• data: Graph data for a single training batch.
• Functionality:
• Performs a forward pass to get graph embeddings.
• Applies an MLP layer and log-softmax activation.
• Computes and returns the cross-entropy loss.
inference_epoch_tj(data):
• Input:
• data: Graph data for a single batch during inference.
• Functionality:
• Similar to train_epoch_tj but used during inference.
• Returns the loss, model outputs, and attention scores.

4.inference(data_loader):
• Input:
• data_loader: DataLoader for inference.
• Functionality:
• Evaluates the model on the provided data loader.
• Collects and returns average loss, true labels, model outputs, predictions,
and node attention scores.
5.evaluate(epoch_idx, data_loader, valid_data_loader):
• Inputs:
• epoch_idx: Current epoch index.
• data_loader: DataLoader for training data.
• valid_data_loader: DataLoader for validation data.

•
Functionality:
• Calls inference on training and validation data.
• Prints and stores metrics, and saves the model if the current test loss is
the minimum.

Detecting Hardware Trojans
We import the required modules and then append the directory of the parent of the script's
directory to the sys.path. This is to ensure that the script can import modules from its parent
directory. An instance of Config is created using command-line arguments (sys.argv[1:]). An
empty list nx_graphs is initialized to store NetworkX graphs. An instance of HW2GRAPH is
created with the provided configuration. The script then processes hardware code from the
specified project path (/home/sharmisthak610/Personal/finalproj/TJ-GNN/driver) into a
graph representation using hw2graph.code2graph. The resulting graph is appended to the
nx_graphs list.
An instance of DataProcessor is created using the configuration (cfg). The script then iterates
over the list of hardware graphs (nx_graphs) and processes each graph using
data_proc.process(hw_graph). After processing the hardware graphs, the script retrieves the
processed graphs using data_proc.get_graphs(). It takes the only processed graph ([0]) and
assigns it to the variable input_data. An instance of the GRAPH2VEC model is created using
the configuration (cfg). The script then specifies a path for the model (model_path), loads the
model configuration (model.cfg), and loads the model weights (model.pth) using
model.load_model(). The model is set to evaluation mode using model.eval(). This is
important when using the model for inference rather than training, as it disables certain
operations like dropout.

This part of the code is performing inference using the previously loaded GRAPH2VEC
model. x is extracted from the processed input data, this represents the node features of the
graph. edge_index is extracted from the processed input data, representing the edge
connections in the graph. The model (GRAPH2VEC) is used for inference by passing the node
features (x), edge connections (edge_index), and batch information (input_data.batch) to
the model. The model output is stored in the variable output. Then, the model output is
passed through a sigmoid function using torch.sigmoid(output[0]) to obtain probability
scores. A threshold of 0.5 is used to convert these probabilities into binary predictions (0 or
1) by thresholding with predictions > threshold.
The resulting binary predictions are then converted to a float tensor using .float() and the
mean of the tensor is calculated using .mean(). This mean represents the average predicted
probability. If the average predicted probability is exactly 1 (with a tolerance for floating-
point imprecision), it prints "Trojan detected." Otherwise, it prints "Trojan not detected."

## Putting Everything Together:
In order to detect hardware trojan, the following steps are followed:
This imports all the necessary files contents and modules. Cfg Creates an instance of the
Config class, passing command-line arguments (sys.argv[1:]) to initialize the configuration
settings. The configuration settings are likely used to control the behaviour of the script,
such as specifying file paths, parameters, or other options.
The above code snippet checks whether the file cfg.data_pkl_path exists. cfg.data_pkl_path
keeps tracks of the pickled elements of the Verilog code, providing a way to serialize and
deserialize Python objects, allowing them to be saved to a file and later reconstructed. If the
file does not exist, the script proceeds to convert graphs using a tool called hw2graph, and
then processes and caches the graph data using a DataProcessor class. If the file already
exists, the script reads the graph data from the cache. In the above code snippet, we first
convert the input hardware code to graph representation, where each hardware design is
transformed into a NetworkX graph representation and then normalized and transformed
into Data instances. If processed Data is already available, then it can be loaded immediately.
Then, in the dataset preparation shown above, we associate each Data instance with a label
corresponding to whether a Trojan exists in the data. Then, we split the entire dataset into
two subsets for training and testing depending on user-defined parameters such as ratio and
seed. These splits are transformed into DataLoader instances so that PyTorch Geometric
utilities can be leveraged.
This code snippet involves the instantiation and configuration of a GRAPH2VEC model based
on the provided configuration (cfg). It creates an instance of the GRAPH2VEC class,
presumably representing a graph-to-vector model, with the configuration cfg. Then it checks
if a pre-trained model path is specified in the configuration (cfg.model_path). If a model
path is provided and the corresponding files ("model.cfg" and "model.pth") exist, it loads the
pre-trained model using model.load_model().
If no pre-trained model path is specified, it configures the model architecture. It then defines
a list of graph convolution layers (convs) with specified parameters (e.g., "gcn" for Graph
Convolutional Network, data_proc.num_node_labels, cfg.hidden). The GNN sets the graph
convolution layers for the model using model.set_graph_conv(convs) and defines a graph
pooling layer (pool) with parameters like "sagpool" for SAGPool and cfg.hidden.
Following this, the GNN sets the graph pooling layer for the model using
model.set_graph_pool(pool) and defines a graph readout layer (readout) with the "max"
readout method. It sets the graph readout layer for the model using
model.set_graph_readout(readout). It defines an output layer (output) as a linear layer with
input size cfg.hidden and output size cfg.embed_dim and sets the output layer for the model
using model.set_output_layer(output).
Now we, move the model to the device specified in the configuration (cfg.device). This is
typically done to utilize GPU acceleration if available. We then, create an instance of the
GraphTrainer class, which is responsible for training and evaluating the graph-to-vector
model and passing the configuration (cfg) and class weights obtained from the training
graphs to the trainer.
trainer.build(model) builds the trainer, involving setting up loss functions, optimizers, and
other training-related configurations. It takes the model (an instance of GRAPH2VEC) as an
argument. We then, initiate the training process using the train_loader for training data and
the valid_loader for validation data and evaluate the trained model based on the specified
number of epochs (cfg.epochs) using both the training and validation data.
We then, create a data loader (vis_loader) for all graphs (for visualization purposes), which
invokes the visualize_embeddings method of the trainer, potentially to generate
embeddings and visualize them. The results are saved in the current directory ("./").

## Output:
We use the following command line input to run the program:
python3.8 detect.py --yaml_path ./config.yaml --raw_dataset_path ../dataset/TJ-RTL-toy --
data_pkl_path dfg_tj_rtl.pkl --graph_type DFG
By this, the command is invoking a Python script called detect.py and providing it with
several command line arguments. These arguments include paths to a YAML configuration
file, a raw dataset, and a data pickle file. Additionally, specification for the type of graph, is
set to "DFG."
The top_module.v code obtained for each of the Verilog file is the input for our program. So,
either the contents could be copied or the path can be changed accordingly.
Now, the results obtained for the following inputs:
1. For RC6 which is Trojan Free
2. PIC16F84-T300 which is Trojan Infected
3. RSS232-T400 which is Trojan Infected
4. VGA which is Trojan Free

## Future scope and improvements:

Reduce Overfitting- Overfitting occurs in machine learning when a model learns the
training data too well, capturing noise or random fluctuations in the data rather than
learning the underlying patterns. As a result, an overfit model perform s well on the
training data but fails to generalize to new, unseen data. In this case, overfitting can be
overcome by increasing the training data.
Optimization Techniques- The goal is to find the optimal set of parameters that result in
the best performance of the model on the given task. This aspect can be improvised on.
Inference Detection- Currently, we are using logistic regression for inference detection,
we can come up with a better and efficient algorithm.
The variance can be reduced.

## Bibliography and References
1.2.3.Tao Han, Yuze Wang, and Peng Liu, “Hardware Trojans Detection at Register Transfer Level
Based on Machine Learning”
Jizhong Yang, Ying Zhang, Member, IEEE, Yifeng Hua, Jiaqi Yao, Zhiming Mao and Xin Chen,
“Hardware Trojans Detection through RTL Features Extraction and Machine Learning”
Rozhin Yasaei*, Shih-Yuan Yu*, Mohammad Abdullah Al Faruque, “GNN4TJ: Graph Neural
Networks for Hardware Trojan Detection at Register Transfer Level”

## Sample Report
![TJ-GNN_page-0001](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/011be0c5-6e9e-4e28-99e0-dec194092502)
![TJ-GNN_page-0002](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/1229ffee-1ecc-4842-917b-0e959baf04d2)
![TJ-GNN_page-0003](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/53aa3953-0cff-463f-b6cb-41cc9c76216c)
![TJ-GNN_page-0004](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/4d40a7ea-bc52-4234-88ae-5c6e8305868c)
![TJ-GNN_page-0005](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/c94f0c2a-89b2-4c1e-8ae7-29766555a580)
![TJ-GNN_page-0006](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/8a698f52-d648-47a7-9003-011b0c6ce38e)
![TJ-GNN_page-0007](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/4febe81f-5c04-4047-8e56-2d774a982b28)
![TJ-GNN_page-0008](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/b57c87a3-7525-4ced-bbe0-28f3c8930271)
![TJ-GNN_page-0009](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/f6a5f038-e50a-48a0-b5a2-98708239347f)
![TJ-GNN_page-0010](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/88021ed2-bd55-47c2-af7f-079e46c2fcd9)
![TJ-GNN_page-0011](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/a1f0a44d-f4ef-4d7d-b50f-2f9139720742)
![TJ-GNN_page-0012](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/21166b36-c40b-49cd-a9f4-a2a4c65b05a2)
![TJ-GNN_page-0013](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/a9ecdc74-4e87-4635-a6f0-7b3701bc67f3)
![TJ-GNN_page-0014](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/f593045c-352f-4ca5-be89-821cdc56714e)
![TJ-GNN_page-0015](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/d7feb544-1069-4d21-904b-b2273af17c78)
![TJ-GNN_page-0016](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/4b673c6e-6704-447c-bae8-456b48be82a9)
![TJ-GNN_page-0017](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/f3d2f7c0-dcb5-4502-83f9-c4a6ee89fd6f)
![TJ-GNN_page-0018](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/46d75532-cc51-4291-aa67-8a3aa76be281)
![TJ-GNN_page-0019](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/a7230a44-4cb5-4505-86ab-af7f64cb828e)
![TJ-GNN_page-0020](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/1408b8ee-bc6c-494e-a794-e58ca1f4571d)
![TJ-GNN_page-0021](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/5320315c-a999-4108-9bd7-430e2360d4f1)
![TJ-GNN_page-0022](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/facb7d31-a732-4f1f-bc3d-89eca2b1c874)
![TJ-GNN_page-0023](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/e1729489-e9c5-4b7b-a34f-267ad493a7a2)
![TJ-GNN_page-0024](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/bb6a94d5-f3c4-4c70-ba3a-f22bbe407d8e)
![TJ-GNN_page-0025](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/60c061c2-4126-4e4b-8043-51f6e5b81048)
![TJ-GNN_page-0026](https://github.com/RitabrataDas343/hardware-trojan-detection-using-GNN/assets/76585827/4cb5a894-9870-4bcf-bbea-18980cd3d131)

