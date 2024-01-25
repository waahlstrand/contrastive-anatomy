import torch
from sklearn import metrics
import numpy as np

import faiss


def test(model, test_loader, train_feature=None, K=5, normalization=True):
    # code is partially adpated from https://github.com/deeplearning-wisc/knn-ood/blob/master/run_cifar.py
    ####KNN###
    if normalization:
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
        normalized_train_feature1 = prepos_feat(train_feature[0]) #number of samples x feature size
        #normalized_train_feature2 = prepos_feat(train_feature[1]) # two augmentations can be applied to training data to create more training feature bank
        
       
    else:
        normalized_train_feature1 =  train_feature[0]
        #normalized_train_feature2 =  train_feature[1]
         
         
    #normalized_train_feature = np.concatenate((normalized_train_feature1, normalized_train_feature2),axis=0)
    normalized_train_feature = normalized_train_feature1 
    index = faiss.IndexFlatL2(normalized_train_feature.shape[1])
    index.add(normalized_train_feature)
    model.eval()
    with torch.no_grad():
        y_score, y_true = [], []
        for i, images in enumerate(test_loader):  
                    x1 = images[0][0].cuda()
                    x2 = images[0][1].cuda()
                    
                    
                    label = images[1]  
                    z1 = model(x1)
                    z2 = model(x2)
                     
                     
                    p1,p2  = model.predictor(z1), model.predictor(z2), 

                    p1= p1.cpu().detach().numpy()
                    p2= p2.cpu().detach().numpy()
                    if normalization:
                        p1 = prepos_feat(p1)
                        p2 = prepos_feat(p2)
                       

                    D1, _ = index.search(p1, K)
                    D2, _ = index.search(p2, K)
                
                    res = np.power(D1[:,-1]*D2[:,-1],1/2)
                    y_true.append(label.cpu())
                    y_score.append(res)
        
        y_true = np.concatenate(y_true)
        y_score = np.array(y_score)
       
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap, y_true, y_score
    