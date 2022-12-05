import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    img_path = "testt.jpg"
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 5
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    import pandas as pd
    df = pd.DataFrame(points, columns = ['X','Y'])
    
    print(df)
    from sklearn.neighbors import NearestNeighbors
    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
    nbrs = NearestNeighbors(n_neighbors=5).fit(df)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(df)
            # sort the neighbor distances (lengths to points) in ascending order
            # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    import matplotlib.pyplot as plt
    k_dist = sort_neigh_dist[:, 4]
            
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=1).fit(df)
    # get cluster labels
    clusters.labels_
    set(clusters.labels_)
    from collections import Counter
    Counter(clusters.labels_)   
    import seaborn as sns
    import matplotlib.pyplot as plt
    prty = sns.scatterplot(data=df, x="X", y="Y", hue=clusters.labels_, legend="full", palette="deep")
    sns.move_legend(prty, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
    plt.show()
    grup=list(set(clusters.labels_))
    points=np.array(points)
    np.save("array.npy",points)
    black = np.zeros((height,width,3))
    print(grup)
    print(clusters.labels_)
    for ik in range(len(grup)):
      afg=np.array(np.random.randint(256, size=3))
      print(afg)
      for jk in range(len(clusters.labels_)):
        if clusters.labels_[jk]==grup[ik]:
          img_to_draw=cv2.circle(black, (int(points[jk,0]),int(points[jk,1])),size, (int(afg[0]),int(afg[1]),int(afg[2])), -1)
          cv2.rectangle(img_to_draw, (5,10), (300, 50), (255,255,200), -1)
          cv2.putText(img_to_draw, f'person detected: {predict_cnt}',(5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (99,93,153), 1)   
    #for p in points:
        #img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image

    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)