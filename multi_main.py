import nni
import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
import numpy as np
from utils.utils import save_json_data, create_dir, load_pkl_data
from common.mbr import MBR
from common.spatial_func import SPoint, distance
from common.road_network import load_rn_shp

from utils.datasets import Dataset, collate_fn
from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from models.multi_train import evaluate, init_weights, train
from models.model import MM_STGED, DecoderMulti, Encoder
from build_graph import load_graph_adj_mtx, load_graph_node_features
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--dataset', type=str, default='Porto',help='data set')
    parser.add_argument('--module_type', type=str, default='simple', help='module type')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--dis_prob_mask_flag', type=bool, default=True, help='flag of using prob mask')
    parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
    parser.add_argument('--online_features_flag', action='store_true', help='flag of using online features')
    parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
    parser.add_argument('--no_attn_flag', type=bool, default=True, help='flag of using attention')
    parser.add_argument('--load_pretrained_flag', default=False, help='flag of load pretrained model')
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--no_debug', type=bool, default=False, help='flag of debug')
    parser.add_argument('--no_train_flag', type=bool, default=True, help='flag of training')
    parser.add_argument('--test_flag', type=bool, default=True, help='flag of testing')
    parser.add_argument('--top_K', type=int, default=10, help='top K value in the decoder')
    parser.add_argument('--RD_inter', type=str, default='1h', help='路况的时间间隔')
    opts = parser.parse_args()

    debug = opts.no_debug
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()
    if opts.dataset == 'Porto':
        args_dict = {
            'module_type':opts.module_type,
            'debug':debug,
            'device':device,

            # pre train
            'load_pretrained_flag':opts.load_pretrained_flag,
            'model_old_path':opts.model_old_path,
            'train_flag':opts.no_train_flag,
            'test_flag':opts.test_flag,

            # attention
            'attn_flag':opts.no_attn_flag,

            # constranit
            'dis_prob_mask_flag':opts.dis_prob_mask_flag,
            'search_dist':50,
            'beta':15,

            # features
            'tandem_fea_flag':opts.tandem_fea_flag,
            'pro_features_flag':opts.pro_features_flag,
            'online_features_flag':opts.online_features_flag,

            # extra info module
            'rid_fea_dim':8,
            'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
            'pro_output_dim':8,
            'poi_num':5,
            'online_dim':5+5,  # poi/roadnetwork features dim
            'poi_type':'company,food,shopping,viewpoint,house',
            'user_num': 442, 
            # MBR
            'min_lat':41.142,
            'min_lng':-8.652,
            'max_lat':41.174,
            'max_lng':-8.578,

            # input data params
            'keep_ratio':opts.keep_ratio,
            'grid_size':opts.grid_size,
            'time_span':15,
            'win_size':25,
            'ds_type':'uniform',
            'split_flag':False,
            'shuffle':True,
            'input_dim':3,

            # model params
            'hid_dim':opts.hid_dim,
            'id_emb_dim':128,
            'dropout':0.5,
            'id_size':2224+1,

            'lambda1':opts.lambda1,
            'n_epochs':opts.epochs,
            'top_K': opts.top_K,
            'batch_size':128,
            'learning_rate':1e-3,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1
        }
    elif opts.dataset == 'Chengdu':
        args_dict = {
            'module_type':opts.module_type,
            'debug':debug,
            'device':device,

            # pre train
            'load_pretrained_flag':opts.load_pretrained_flag,
            'model_old_path':opts.model_old_path,
            'train_flag':opts.no_train_flag,
            'test_flag':opts.test_flag,

            # attention
            'attn_flag':opts.no_attn_flag,

            # constranit
            'dis_prob_mask_flag':opts.dis_prob_mask_flag,
            'search_dist':50,
            'beta':15,

            # features
            'tandem_fea_flag':opts.tandem_fea_flag,
            'pro_features_flag':opts.pro_features_flag,
            'online_features_flag':opts.online_features_flag,

            # extra info module
            'rid_fea_dim':8,
            'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
            'pro_output_dim':8,
            'poi_num':5,
            'online_dim':5+5,  # poi/roadnetwork features dim
            'poi_type':'company,food,shopping,viewpoint,house',
            'user_num': 17675,

            # MBR
            'min_lat':30.655,
            'min_lng':104.043,
            'max_lat':30.727,
            'max_lng':104.129,

            # input data params
            'keep_ratio':opts.keep_ratio,
            'grid_size':opts.grid_size,
            'time_span':15,
            'win_size':25,
            'ds_type':'uniform',
            'split_flag':False,
            'shuffle':True,
            'input_dim':3,

            # model params
            'hid_dim':opts.hid_dim,
            'id_emb_dim':128,
            'dropout':0.5,
            'id_size':2504+1,

            'lambda1':opts.lambda1,
            'n_epochs':opts.epochs,
            'top_K': opts.top_K,
            'RD_inter': opts.RD_inter,
            'batch_size':128,
            'learning_rate':1e-3,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1
        }
    
    assert opts.dataset in ['Porto', 'Chengdu'], 'Check dataset name if in [Porto, Chengdu]'

    args.update(args_dict)

    print('Preparing data...')

    train_trajs_dir = "./data/{}/train/".format(opts.dataset)
    valid_trajs_dir = "./data/{}/valid/".format(opts.dataset)
    test_trajs_dir = "./data/{}/test/".format(opts.dataset)

    extra_info_dir = "./data/{}/extra_data/".format(opts.dataset)
    rn_dir = extra_info_dir + "road_network/"
    user_dir = json.load(open( extra_info_dir + "uid2index.json"))
    SE_file = extra_info_dir + '{}_SE.txt'.format(opts.dataset)
    condition_file = extra_info_dir + 'flow.npy'
    road_file = extra_info_dir + 'TLG/graph_A.csv'
    
    if args.tandem_fea_flag:
        fea_flag = True
    else:
        fea_flag = False

    model_save_path = './results/'+args.module_type+'_kr_'+str(args.keep_ratio)+'_debug_'+str(args.debug)+\
        '_gs_'+str(args.grid_size)+'_lam_'+str(args.lambda1)+\
        '_attn_'+str(args.attn_flag)+'_prob_'+str(args.dis_prob_mask_flag)+\
        '_fea_'+str(fea_flag)+'_'+time.strftime("%Y%m%d_%H%M%S") + '/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a+')
    # spatial embedding
    spatial_A = load_graph_adj_mtx(road_file)
    spatial_A_trans = np.zeros((spatial_A.shape[0]+1, spatial_A.shape[1]+1)) + 1e-10
    spatial_A_trans[1:,1:] = spatial_A

    road_condition = np.load(condition_file) # T, N, N
    for i in range(road_condition.shape[0]):
        maxn = road_condition[i].max()
        road_condition[i] = road_condition[i] / maxn

    f = open(SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0])+1, int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index+1] = temp[1 :]
    
    SE = torch.from_numpy(SE)

    rn = load_rn_shp(rn_dir, is_directed=True)
    raw_rn_dict = load_rn_dict(extra_info_dir, file_name='raw_rn_dict.json')
    new2raw_rid_dict = load_rid_freqs(extra_info_dir, file_name='new2raw_rid.json')
    raw2new_rid_dict = load_rid_freqs(extra_info_dir, file_name='raw2new_rid.json')
    rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json')

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    args_dict['max_xid'] = max_xid
    args_dict['max_yid'] = max_yid
    args.update(args_dict)
    print(args)
    logging.info(args_dict)
    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write(str(args_dict))
        f.write('\n')
    # load features
    weather_dict = None #load_pkl_data(extra_info_dir, 'weather_dict.pkl')
    if args.online_features_flag:
        grid_poi_df = pd.read_csv(extra_info_dir+'poi'+str(args.grid_size)+'.csv',index_col=[0,1])
        norm_grid_poi_dict = get_poi_info(grid_poi_df, args)
        norm_grid_rnfea_dict = get_rn_info(rn, mbr, args.grid_size, grid_rn_dict, rn_dict)
        online_features_dict = get_online_info_dict(grid_rn_dict, norm_grid_poi_dict, norm_grid_rnfea_dict, args)
    else:
        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
    rid_features_dict = None

    # load dataset
    train_dataset = Dataset(train_trajs_dir, user_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                            norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                            parameters=args, debug=debug)
    valid_dataset = Dataset(valid_trajs_dir, user_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                            norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                            parameters=args, debug=debug)
    test_dataset = Dataset(test_trajs_dir, user_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                           norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                           parameters=args, debug=debug)
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('test dataset shape: ' + str(len(test_dataset)))

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=collate_fn,
                                                num_workers=4, pin_memory=False)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=collate_fn,
                                                num_workers=4, pin_memory=False)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=args.shuffle, collate_fn=collate_fn,
                                               num_workers=4, pin_memory=False)

    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('test dataset shape: ' + str(len(test_dataset)))

    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write('Finish data preparing.' + '\n')
        f.write('training dataset shape: ' + str(len(train_dataset)) + '\n')
        f.write('validation dataset shape: ' + str(len(valid_dataset)) + '\n')
        f.write('test dataset shape: ' + str(len(test_dataset)) + '\n')


    enc = Encoder(args)
    dec = DecoderMulti(args)
    model = MM_STGED(enc, dec, args.hid_dim, args.max_xid, args.max_yid, args.top_K).to(device)
    model.apply(init_weights)  # learn how to init weights
    if args.load_pretrained_flag:
        model.load_state_dict(torch.load(args.model_old_path + 'val-best-model.pt'))

    print('model', str(model))
    logging.info('model' + str(model))
    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write('model' + str(model) + '\n')

    if args.train_flag:
        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
        ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

        dict_train_loss = {}
        dict_valid_loss = {}
        best_valid_loss = float('inf')  # compare id loss

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        for epoch in tqdm(range(args.n_epochs)):
            start_time = time.time()

            new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, \
            train_rate_loss, train_id_loss = train(model, spatial_A_trans, road_condition, SE, train_iterator, optimizer, log_vars,
                                                   rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
                                                   online_features_dict, rid_features_dict, args)

            valid_id_acc1, valid_id_recall, valid_id_precision, valid_dis_mae_loss, valid_dis_rmse_loss, \
            valid_dis_rn_mae_loss, valid_dis_rn_rmse_loss, \
            valid_rate_loss, valid_id_loss = evaluate(model, spatial_A_trans, road_condition, SE, valid_iterator,
                                                      rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
                                                      online_features_dict, rid_features_dict, raw_rn_dict,
                                                      new2raw_rid_dict, args)
            ls_train_loss.append(train_loss)
            ls_train_id_acc1.append(train_id_acc1)
            ls_train_id_recall.append(train_id_recall)
            ls_train_id_precision.append(train_id_precision)
            ls_train_rate_loss.append(train_rate_loss)
            ls_train_id_loss.append(train_id_loss)

            ls_valid_id_acc1.append(valid_id_acc1)
            ls_valid_id_recall.append(valid_id_recall)
            ls_valid_id_precision.append(valid_id_precision)
            ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
            ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
            ls_valid_dis_rn_mae_loss.append(valid_dis_rn_mae_loss)
            ls_valid_dis_rn_rmse_loss.append(valid_dis_rn_rmse_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            ls_valid_id_loss.append(valid_id_loss)
            valid_loss = valid_rate_loss + valid_id_loss
            ls_valid_loss.append(valid_loss)

            dict_train_loss['train_ttl_loss'] = ls_train_loss
            dict_train_loss['train_id_acc1'] = ls_train_id_acc1
            dict_train_loss['train_id_recall'] = ls_train_id_recall
            dict_train_loss['train_id_precision'] = ls_train_id_precision
            dict_train_loss['train_rate_loss'] = ls_train_rate_loss
            dict_train_loss['train_id_loss'] = ls_train_id_loss

            dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
            dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
            dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
            dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
            dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
            dict_valid_loss['valid_dis_mae_loss'] = ls_valid_dis_mae_loss
            dict_valid_loss['valid_dis_rmse_loss'] = ls_valid_dis_rmse_loss
            dict_valid_loss['valid_dis_rn_mae_loss'] = ls_valid_dis_rn_mae_loss
            dict_valid_loss['valid_dis_rn_rmse_loss'] = ls_valid_dis_rn_rmse_loss
            dict_valid_loss['valid_id_loss'] = ls_valid_id_loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')

            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
                logging.info('log_vars:' + str(weights))
                logging.info('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss))
                logging.info('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss))
                with open(model_save_path+'logging.txt', 'a+') as f:
                    f.write('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
                    f.write('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss) + 
                             '\n')
                    f.write('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss) + 
                             '\n')
                    f.write('\n')
                torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
                save_json_data(dict_train_loss, model_save_path, "train_loss.json")
                save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")

    if args.test_flag:
        model.load_state_dict(torch.load(model_save_path + 'val-best-model.pt'))
        start_time = time.time()
        test_id_acc1, test_id_recall, test_id_precision, test_dis_mae_loss, test_dis_rmse_loss, \
        test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss = evaluate(model, spatial_A_trans, road_condition, SE, test_iterator,
                                                                                             rn_dict, grid_rn_dict, rn,
                                                                                             raw2new_rid_dict,
                                                                                             online_features_dict,
                                                                                             rid_features_dict,
                                                                                             raw_rn_dict, new2raw_rid_dict,
                                                                                             args)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        logging.info('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        logging.info('\tTest RID Acc1:' + str(test_id_acc1) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                     '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                     '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                     '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                     '\tTest Rate Loss:' + str(test_rate_loss) +
                     '\tTest RID Loss:' + str(test_id_loss))
        
        with open(model_save_path+'logging.txt', 'a+') as f:
            f.write("\n")
            f.write('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
            f.write('\tTest RID Acc1:' + str(test_id_acc1) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                     '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                     '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                     '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                     '\tTest Rate Loss:' + str(test_rate_loss) +
                     '\tTest RID Loss:' + str(test_id_loss) +
                     '\n')