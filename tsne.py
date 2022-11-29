import torch
import torch.nn as nn
import colorsys
import pandas as pd
from typing import Iterable, Dict, List, Tuple, Callable
from torch import Tensor
from IPython.display import display_html 
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# some functions from sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
# Do what you need in order to transform the output of the model into a sequence of
# predictions
# example, most models output a [batchsize,num_classes] and don't do argmax
# you need to do the argmax here
# expect as output a tensor of shape [batchsize,]
# the example here works for a model that outputs as first item of a list [batchsize,num_classes]
# and the other satuff is stuff that was used in training and I don't need here
def callback(input:Tensor) -> Tensor:
    output = torch.argmax(input[0],axis=1)
    return output
# this one works on most models
def callback_sample(input:Tensor) -> Tensor:
    output = torch.argmax(input,axis=1)#
    return output
# How to use
# Your Model
# source data loader
# target data loader
# class mapping. Example {'backpack':0,'purse':1, ...} etc. Please use the same of the training
# callback= required if your model doesn't output a prediction in the shape [batch,] but in [batch,num_classes]
# check cell below to see example
def eval_plot_source_target(model: nn.Module,
                            source_loader: Iterable,
                            target_loader: Iterable,
                            class_mapping: Dict[str,int],
                            device: str ='cpu',callback_fn=None, colormap:Dict[int,Tuple[int,int,int]] = None ) -> None:
    
    class Extractor(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = nn.Sequential(*(list(model.children())[:-1]))
            self.cls =  list(model.children())[-1]
        def forward(self, x: Tensor) -> Tensor:
            out = self.model(x)
            out = out.squeeze()
            out = out.reshape(x.shape[0],-1)
            return out
    def ref_labels(loader: Iterable) -> Iterable:
        data = []
        for i in loader:
            data.append(i[1])
        return np.concatenate(data,axis=0)
    def color_map(class_labels: Dict[str,int]) -> Dict[int,Tuple[int,int,int]]:
        colormap = {}
        for k,i in class_labels.items():
            i = i/len(class_labels)
            (h, s, v) = (0.0+i, 0.5, 1)
            (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
            colormap[int (i * len(class_labels))] = (round(r,2),round(g,2),round(b,2))
        return colormap
    def color(labels: Iterable[str], colormap: Dict[int,Tuple[int,int,int]]) -> Iterable[Tuple[int,int,int]]:
        colors = []
        for i in labels:
            colors.append(colormap[i])
        return colors

    extractor = Extractor(model).eval().to(device)
    model.to(device)
    if colormap is None:
        colormap = color_map(class_mapping)
    else:  
        colormap = colormap
    source_features = []
    for input,prediction in source_loader:
        e = extractor(input.to(device))
        source_features.append(e.cpu().detach().numpy())

    source_features = np.concatenate(source_features,axis=0)

    source_out=[]
    for input,prediction in source_loader:
        e = model(input.to(device))
        if callback_fn!=None:
            e = callback_fn(e)
        e = e.cpu().detach().numpy()
        source_out.append(e)
    source_out = np.concatenate(source_out,axis=0)

    ref_source = ref_labels(source_loader)
   

    target_features = []
    for input,prediction in target_loader:
        e = extractor(input.to(device))
        target_features.append(e.cpu().detach().numpy())
    target_features = np.concatenate(target_features,axis=0)

    target_out=[]
    for input,prediction in target_loader:
        e = model(input.to(device))
        if callback_fn!=None:
            e = callback_fn(e)
        e = e.cpu().detach().numpy()
        target_out.append(e)
    target_out = np.concatenate(target_out,axis=0)

    ref_target = ref_labels(target_loader)
   

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))


    tsne = TSNE(n_components=2,  learning_rate='auto', init='pca')
    tsne_out = tsne.fit_transform(np.concatenate((source_features,target_features),axis=0))
    tsne_out_source,tsne_out_target = tsne_out[:len(source_features)],tsne_out[-len(target_features):]

    # source
    # get ref labels
    source_ref_colors = color(ref_source,colormap)
    # get hyp labels
    source_hyp_colors = color(source_out,colormap)
    # compute TSNE
    
    # plot features
    ax1.scatter(tsne_out_source[:,0], tsne_out_source[:,1],c=source_ref_colors)
    ax1.set_title('Reference Embedding')
    # plot features
    ax2.scatter(tsne_out_source[:,0], tsne_out_source[:,1],c=source_hyp_colors)
    ax2.set_title('Hypothesis Embedding')

    
    # target
    # get ref labels
    target_ref_colors = color(ref_target,colormap)
    # get hyp labels
    target_hyp_colors = color(target_out,colormap)
    # compute TSNE
    #tsne = TSNE(n_components=2,  learning_rate='auto', init='pca')
    #tsne_out = tsne.fit_transform(target_features)
    # plot features
    ax3.scatter(tsne_out_target[:,0], tsne_out_target[:,1],c=target_ref_colors)
    ax3.set_title('Reference Embedding')
    # plot features
    ax4.scatter(tsne_out_target[:,0], tsne_out_target[:,1],c=target_hyp_colors)
    ax4.set_title('Hypothesis Embedding')

    plt.figtext(-0.25, 0.75, 'Source', fontsize = 14)
    plt.figtext(-0.25, 0.25, 'Target', fontsize = 14)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for idx,color in colormap.items()]
    inv_map = {v: k for k, v in class_mapping.items()}
    plt.legend(markers, [inv_map[k] for k,v in colormap.items()], numpoints=1,loc=(1.25,0))
    
    fig.suptitle("TSNE of the embeddings", fontsize=14)
    plt.show()

        # classification report
    names = [k for k,v in class_mapping.items()]

    report_source = classification_report(ref_source,source_out,target_names=names,output_dict=True)
    df_source = pd.DataFrame(report_source).transpose()
    df_source=df_source.round(2)

    report_target = classification_report(ref_target,target_out,target_names=names,output_dict=True)
    df_target = pd.DataFrame(report_target).transpose()
    df_target = df_target.round(2)

    df1_styler = df_source.style.set_table_attributes("style='display:inline'").set_caption('source').set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-style', 'italic'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
                ]
            }])
    
    df2_styler = df_target.style.set_table_attributes("style='display:inline'").set_caption('target').set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-style', 'italic'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
                ]
            }])
    space = "\xa0" * 10
    display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)
