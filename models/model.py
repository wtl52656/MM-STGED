import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy

def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Edge_merge(nn.Module):
    def __init__(self, node_in_channel, edge_in_channel, out_channel):
        super(Edge_merge, self).__init__()
        self.Z = nn.Linear(edge_in_channel, out_channel)
        self.H = nn.Linear(node_in_channel, out_channel)
    
    def forward(self, edge, node):
        # edge: B,T,T,2
        # node: B,T,F
        edge_transform = self.Z(edge)  # B,T,T,F

        node_transform = self.H(node)  # B,T,F
        node_transform_i = node_transform.unsqueeze(2) #B,1,T,F
        node_transform_j = node_transform.unsqueeze(1) #B,T,1,F
        return edge_transform + node_transform_i + node_transform_j  # B,T,T,F

class my_GCN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(my_GCN, self).__init__()
        self.linear1 = nn.Linear(in_channel, out_channel, bias=False).to('cuda:0')
        self.linear2 = nn.Linear(in_channel, out_channel, bias=False).to('cuda:0')
        
        self.wh = nn.Linear(out_channel, out_channel, bias=False).to("cuda:0")
        self.wtime = nn.Linear(out_channel, out_channel, bias=False).to("cuda:0")
        self.wloca = nn.Linear(out_channel, out_channel, bias=False).to("cuda:0")
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.edge_merge = Edge_merge(in_channel, 2, out_channel)
        self.w_edge = nn.Linear(out_channel, out_channel, bias=False).to("cuda:0")
    def forward(self, X, A1, A2):
        # X: B,T,F
        # A: B,T,T
        A1X = torch.bmm(A1, X)
        AXW1 = self.relu(self.linear1(A1X))

        A2X = torch.bmm(A2, X)
        AXW2 = self.relu(self.linear2(A2X))

        A_merge = torch.cat((A1.unsqueeze(-1), A2.unsqueeze(-1)),dim=-1)  # B,T,T,2
        _edge_merge = self.edge_merge(A_merge, X)  #B,T,T,F
        _edge_merge = torch.sum(_edge_merge, dim=2) #B,T,F
        _merge = self.wh(X) + self.wtime(AXW1) + self.wloca(AXW2) + self.w_edge(_edge_merge)
        
        
        norm = self.bn(_merge.permute(0,2,1)).permute(0,2,1)
        all_state = X + self.relu(norm)
        hidden = torch.mean(all_state, 1).unsqueeze(0)
        return all_state.permute(1,0,2), hidden

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.pro_features_flag = parameters.pro_features_flag

        input_dim = parameters.input_dim 
        self.input_cat = nn.Linear(64*2, parameters.hid_dim, bias = False)
        self.relu = nn.ReLU(inplace=True)
        if self.online_features_flag:
            input_dim = input_dim + parameters.online_dim

        self.rnn = nn.GRU(input_dim, self.hid_dim)
        self.dropout = nn.Dropout(parameters.dropout)

        if self.pro_features_flag:
            self.extra = Extra_MLP(parameters)
            self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)

    def forward(self, src, src_len, pro_features):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        # hidden = [1, batch size, hidden_dim]
        # outputs = [src len, batch size, hidden_dim * num directions]
            
        if self.pro_features_flag:
            extra_emb = self.extra(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            # extra_emb = [1, batch size, extra output dim]
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
            # hidden = [1, batch size, hid dim]

        return outputs, hidden



class attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(attention, self).__init__()
        self.l_k = nn.Linear(in_channel, out_channel)
        self.l_q = nn.Linear(in_channel, out_channel)
        self.l_v = nn.Linear(in_channel, out_channel)
    def forward(self, x_k, x_q, mask=None, dropout=None):
        key = self.l_k(x_k)
        query = self.l_q(x_q)
        value = self.l_v(x_k)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # p_attn: (batch, N, h, T1, T2)

        return torch.matmul(p_attn, value)  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)



class Attention(nn.Module):
    # TODO update to more advanced attention layer.
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)


class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        self.emb_id = nn.Embedding(self.id_size, self.id_emb_dim)
        self.top_k = parameters.top_K
        rnn_input_dim = self.id_emb_dim + 1 + 64
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim
        
        type_input_dim = self.id_emb_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
                          nn.Linear(type_input_dim, self.hid_dim),
                          nn.ReLU()
                          )
        self.user_embedding = nn.Embedding(parameters.user_num, 10)      
        self.user_merge_layer = nn.Sequential(
            nn.Linear(self.hid_dim + 10 + 64, self.hid_dim)
        )
        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim 

        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network
            
        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim
        
        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)
        
        
    def forward(self, decoder_node2vec, user_id, road_index, spatial_A_trans, topk_mask, trg_index, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                pre_grid, next_grid, constraint_vec, pro_features, online_features, rid_features):

        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        # input_id = [1, batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        embedded = self.dropout(self.emb_id(input_id))
        # embedded = [1, batch size, emb dim]

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate, 
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)
        rnn_input = torch.cat((rnn_input, road_index.unsqueeze(0)), dim=2)
        
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        user_info = self.user_embedding(user_id)
        tra_vec = torch.mean(decoder_node2vec, dim=1).unsqueeze(0)
        user_merge = self.user_merge_layer(torch.cat((output, user_info, tra_vec), dim=2))

        if self.dis_prob_mask_flag:
            if topk_mask is not None:
                trg_index_repeat = trg_index.repeat(1, constraint_vec.shape[1])
                _tmp_mask = 0
                for i in range(self.top_k):
                    id_index = topk_mask[:,i:i+1].squeeze(1).long()
                    _tmp_mask = _tmp_mask + spatial_A_trans[id_index]
                _tmp_mask[_tmp_mask>1] = 1.
                constraint_vec = torch.where(trg_index_repeat==1, constraint_vec, _tmp_mask)
                # print(constraint_vec.shape)
                # exit()
            prediction_id = mask_log_softmax(self.fc_id_out(user_merge.squeeze(0)), 
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(user_merge.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, user_merge.squeeze(0)),dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden


class spatialTemporalConv(nn.Module):
    def __init__(self, in_channel, base_channel):
        super(spatialTemporalConv, self).__init__()
        self.start_conv = nn.Conv2d(in_channel, base_channel, 1, 1, 0)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channel)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, road_condition):
        #road_condition: T, N, N
        T, N, N = road_condition.shape
        # print(road_condition.unsqueeze(-1).shape)
        _start = self.start_conv(road_condition.unsqueeze(1))  # T,1,N,N
        spatialConv = self.spatial_conv(_start)  #T,F,N,N
        spatial_reshape = spatialConv.reshape(T, -1, N*N).permute(2, 1, 0) # N*N,F,T
        temporalConv = self.temporal_conv(spatial_reshape)
        conv_res = temporalConv.reshape(N, N, -1, T).permute(3, 2, 0, 1)  # T,F,N,N
        # print((_start + conv_res).shape)
        return (_start + conv_res).permute(0, 2, 3, 1)



class MM_STGED(nn.Module):
    def __init__(self, encoder, decoder, base_channel, x_id, y_id, top_k):
        super(MM_STGED, self).__init__()
        self.encoder = encoder  # Encoder
        self.decoder = decoder
        self.spatialTemporalConv = spatialTemporalConv(1, 64)
        
        self.dropout = nn.Dropout(p=0.3)
        self.x_id = x_id
        self.y_id = y_id
        
        self.topK = top_k
        self.fc_rate_out = nn.Sequential(
            nn.Linear(base_channel + base_channel, base_channel),
            nn.ReLU(inplace=True),
            nn.Linear(base_channel, 1)
        )
        self.encoder_out = nn.Sequential(
            nn.Linear(512+64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.encoder_point_cat = nn.Sequential(
            nn.Linear(512+64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.mygcn = my_GCN(base_channel, base_channel)
        self.device = "cuda:0"

    def forward(self, user_tf_idf, spatial_A_trans, road_condition, src_road_index_seqs, SE, tra_time_A,tra_loca_A,  src_len, src_grid_seqs, src_eid_seqs, src_rate_seqs, trg_in_grid_seqs,
                trg_in_index_seqs, trg_rids, trg_rates, trg_len,
                pre_grids, next_grids, constraint_mat, pro_features, 
                online_features_dict, rid_features_dict,
                teacher_forcing_ratio=0.5):
                
        batchsize, max_src_len, _ = src_grid_seqs.shape
        
        """Graph-based trajectory encoder"""
        src_attention, src_hidden = self.encoder(src_grid_seqs.permute(1,0,2), src_len, pro_features)
        src_attention = src_attention.permute(1, 0, 2)   # B, T, F
        src_attention, src_hidden = self.mygcn(src_attention, tra_time_A,tra_loca_A)

        """add road """
        # _trg_in_index_seqs = trg_in_index_seqs.repeat(1, 1, constraint_mat.shape[2])
        # print(constraint_mat.shape, src_grid_seqs.shape, SE.shape, trg_in_index_seqs.shape)

        # tmp_constraint_mat = constraint_mat.permute(1, 0, 2)
        # tmp_constraint_mat = torch.where(_trg_in_index_seqs==1., tmp_constraint_mat, 0)

        # for batch in range(batchsize):
        #     for time_l in range(constraint_mat.shape[0]):
        #         if trg_in_index_seqs[batch, time_l, 0] == 1:  #观测到的节点 
        #             tmp_road_dis = constraint_mat[time_l][batch]
        #             summ_road = torch.
        #             print(tmp_road_dis.max())
                    
        #             exit()
        all_road_embed =  torch.einsum('btr,rd->btd',(constraint_mat.permute(1, 0, 2), SE))   # B, T, F
        summ = constraint_mat.permute(1, 0, 2).sum(-1).unsqueeze(-1)
        trajectory_point_embed = all_road_embed / summ   #  得到了每个节点的表示
        

        trajectory_point_sum = trg_in_index_seqs.sum(1)
        trajectory_embed = (trajectory_point_embed.sum(1) / trajectory_point_sum).unsqueeze(0)  # 得到每个轨迹的表示
        src_hidden = self.encoder_out(torch.cat((src_hidden, trajectory_embed), -1))   # 轨迹最终表示：路段表示+原始图表示

        # 接下来基于节点的表示trajectory_point_embed，将其拼接到GRU输出上
        _trg_in_index_seqs = trg_in_index_seqs.repeat(1, 1, 64)
        _imput_zero = torch.zeros((batchsize, 64)).to("cuda:0")
        trajectory_point_road = []

        # _traject_i_index = trg_in_index_seqs.repeat(1, 1, 64)
        # b = trajectory_point_embed[_traject_i_index==1.]
        # print(b.shape, trg_in_index_seqs.sum())
        # exit()


        trajectory_point_road = torch.zeros((max_src_len, batchsize, 64)).to("cuda:0")
        # dimension = len(sequence[0])
        # start = [0] * dimension  # pad 0 as start of rate sequence
        # new_sequence.append(start)
        # new_sequence.extend(sequence)
        # new_sequence = torch.tensor(new_sequence)


        for batch in range(batchsize):
            # print(src_grid_seqs[0])
            _traject_i_index = trg_in_index_seqs[batch,:, 0]
            # print(_traject_i_index, _traject_i_index.sum())
            b = trajectory_point_embed[batch][_traject_i_index==1.]
            curr_traject_i_length = b.shape[0]
            # print(b.shape, '....', trajectory_point_road[1:1+curr_traject_i_length, batch].shape, '....', trajectory_point_road.shape)
            trajectory_point_road[1:1+curr_traject_i_length, batch] = b
            
        # exit()
        # for time in range(max_src_len):
        #     tmp_road_embed = torch.where(_trg_in_index_seqs[:,time,]==1., trajectory_point_embed[:, time,], _imput_zero).unsqueeze(0)  # 判断当前点是不是观测点，如果是，则拼到GRU后面
        #     if time == 0:
        #         trajectory_point_road = tmp_road_embed
        #     else:
        #         trajectory_point_road = torch.cat((trajectory_point_road, tmp_road_embed), 0)
        
        src_attention = self.encoder_point_cat(torch.cat((src_attention, trajectory_point_road), -1))



        # contextual road condition representation
        road_conv = self.spatialTemporalConv(road_condition)  # T, N, N, F

        #trajectory-related road condition
        road_index = None
        for i in range(1, max_src_len):  #ignore the first point
            TNN_i = src_road_index_seqs[:,i]  # B,3
            _tmp = road_conv[TNN_i[:,0], TNN_i[:,1], TNN_i[:,2]].unsqueeze(1)    # B,T, F
            if i == 1:
                road_index = _tmp
            else:
                road_index = torch.cat((road_index, _tmp), 1)
        road_index = road_index.mean(1)
        
        
        max_trg_len = trg_rids.size(0)
        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batchsize, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None

        outputs_id, outputs_rate = self.normal_step(SE, user_tf_idf, road_index, spatial_A_trans, trg_in_index_seqs, max_trg_len, batchsize, trg_rids, trg_rates, trg_len,
                                                    src_attention, src_hidden, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    pre_grids, next_grids, constraint_mat, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate

    def normal_step(self, SE, user_tf_idf, road_index, spatial_A_trans, trg_in_index_seqs, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    pre_grids, next_grids, constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        
        decoder_node2vec = SE[input_id.long()]
        topk_mask = None
        for t in range(1, max_trg_len):
            trg_index = trg_in_index_seqs[:,t]   #batchsize 
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            prediction_id, prediction_rate, hidden = self.decoder(decoder_node2vec, user_tf_idf, road_index, spatial_A_trans, topk_mask, trg_index, input_id, input_rate, hidden, encoder_outputs,
                                                                     attn_mask, pre_grids[t], next_grids[t],
                                                                     constraint_mat[t], pro_features, online_features,
                                                                     rid_features)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

            #topK mask
            topk_mask = prediction_id.topk(self.topK, dim=-1, sorted=True)[1]
            
            decoder_node2vec = torch.cat((decoder_node2vec, SE[input_id.long()]),dim=1)

        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1
        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = 0
            outputs_id[i][trg_len[i]:, 0] = 1  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)
        return outputs_id, outputs_rate