import torch.nn as nn
from torch.nn import init

from models.Treelstm import basic
from models.Treelstm.utils import *

class AttnCombiner(nn.Module):
    def __init__(self, bidirectional, hidden_size, attention_size=128, dropout=0):
        super(AttnCombiner, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.ws1 = nn.Linear(hidden_size * self.num_directions, attention_size, bias=False)
        self.ws2 = nn.Linear(attention_size, 1, bias=False)
        init.orthogonal(self.ws1.weight.data)
        init.orthogonal(self.ws2.weight.data)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, hiddens):
        size = hiddens.size()  # [bsz, len, in_dim]
        x_flat = hiddens.contiguous().view(-1, size[2])  # [bsz*len, in_dim]
        h_bar = self.tanh(self.ws1(self.drop(x_flat)))  # [bsz*len, attn_hid]
        alphas = self.ws2(h_bar).view(size[0], size[1])  # [bsz, len]
        return alphas

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

    def forward(self, left=None, right=None):
        """
        Args:
            left: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            right: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = left
        hr, cr = right
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = basic.apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TreeLSTMEncoder(nn.Module):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional):
        super(TreeLSTMEncoder, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.pooling_method = pooling_method
        self.bidirectional = bidirectional

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim * 2 if self.bidirectional else hidden_dim)

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal(self.leaf_rnn_cell.weight_hh.data)
            init.constant(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal(self.word_linear.weight.data)
            init.constant(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2).expand_as(new_h)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def forward(self, *inp):
        pass


class GumbelTreeLSTMEncoder(TreeLSTMEncoder):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional, gumbel_temperature):
        super(GumbelTreeLSTMEncoder, self).__init__(
            word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional)
        self.gumbel_temperature = gumbel_temperature
        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim * 2 if self.bidirectional else hidden_dim))
        if pooling_method == 'attention':
            self.combiner = AttnCombiner(bidirectional, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        super(GumbelTreeLSTMEncoder, self).reset_parameters()
        init.normal(self.comp_query.data, mean=0, std=0.01)

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = basic.dot_nd(query=self.comp_query, candidates=new_h)
        if self.training:
            select_mask = basic.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature, mask=mask)
        else:
            select_mask = basic.greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, inp, length, return_select_masks=False):
        max_depth = inp.size(1)
        length_mask = basic.sequence_mask(sequence_length=length, max_length=max_depth)
        select_masks = list()
        features = list()

        if self.use_leaf_rnn:
            hs = list()
            cs = list()
            batch_size, max_length, _ = inp.size()
            zero_state = Variable(inp.data.new(batch_size, self.hidden_dim)
                                  .zero_())
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=inp[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = list()
                cs_bw = list()
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=inp, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = basic.apply_nd(fn=self.word_linear, input=inp)
            state = state.chunk(chunks=2, dim=2)
        nodes = list()
        if self.pooling_method is not None:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            left = (h[:, :-1, :], c[:, :-1, :])
            right = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(left=left, right=right)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i+1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                features.append(selected_h)
                if self.pooling_method is not None:
                    nodes.append(selected_h.unsqueeze(1))
            done_mask = length_mask[:, i+1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if (self.pooling_method is not None) and i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.pooling_method == 'max':
            nodes = torch.cat(nodes, dim=1)
            h = nodes.max(1)[0].unsqueeze(1)
        elif self.pooling_method == 'mean':
            nodes = torch.cat(nodes, dim=1).sum(1)
            lengths = length * 2 - 1
            lengths = lengths.unsqueeze(1).float().expand_as(nodes)
            h = (nodes / lengths).unsqueeze(1)
        elif self.pooling_method == 'attention':
            nodes = torch.cat(nodes, dim=1)
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask_expand = att_mask.float().unsqueeze(2).expand_as(nodes)
            att_weights = basic.masked_softmax(
                logits=self.combiner(nodes), mask=att_mask)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            h = (att_weights_expand * att_mask_expand * nodes).sum(1).unsqueeze(1)
        else:
            assert self.pooling_method is None
        assert h.size(1) == 1 and c.size(1) == 1
        if not return_select_masks:
            return h.squeeze(1), c.squeeze(1)
        else:
            return h.squeeze(1), c.squeeze(1), features, select_masks


class RecursiveTreeLSTMEncoder(TreeLSTMEncoder):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional):
        super(RecursiveTreeLSTMEncoder, self).__init__(
            word_dim, hidden_dim, use_leaf_rnn, pooling_method, bidirectional)
        self.reset_parameters()
        self.pooling_method = pooling_method
        if self.pooling_method == 'attention':
            self.combiner = AttnCombiner(bidirectional, hidden_dim)

    def forward(self, inp, length, fixed_masks):
        # print("="*100)
        # print(inp.shape)
        # print(length)
        # print(fixed_masks)
        max_depth = inp.size(1)
        length_mask = basic.sequence_mask(sequence_length=length, max_length=max_depth)
        features = list()

        if self.use_leaf_rnn:
            hs = list()
            cs = list()
            batch_size, max_length, _ = inp.size()
            zero_state = Variable(inp.data.new(batch_size, self.hidden_dim).zero_())
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=inp[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = list()
                cs_bw = list()
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=inp, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = basic.apply_nd(fn=self.word_linear, input=inp)
            state = state.chunk(chunks=2, dim=2)
        nodes = list()
        if self.pooling_method is not None:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            left = (h[:, :-1, :], c[:, :-1, :])
            right = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(left=left, right=right)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, selected_h = self.compose(
                    old_state=state, new_state=new_state, tree_mask=fixed_masks[i])
                new_state = (new_h, new_c)
                features.append(selected_h)
                if self.pooling_method is not None:
                    nodes.append(selected_h.unsqueeze(1))
            done_mask = length_mask[:, i + 1]
            state = self.update_state(
                old_state=state, new_state=new_state, done_mask=done_mask)
            if (self.pooling_method is not None) and i >= max_depth - 2:
                done_masks = done_mask.float().unsqueeze(1).unsqueeze(2).expand_as(state[0])
                nodes.append(state[0] * done_masks)
        h, c = state
        if self.pooling_method == 'max':
            nodes = torch.cat(nodes, dim=1)
            h = nodes.max(1)[0].unsqueeze(1)
        elif self.pooling_method == 'mean':
            nodes = torch.cat(nodes, dim=1).sum(1)
            lengths = length * 2 - 1
            lengths = lengths.unsqueeze(1).float().expand_as(nodes)
            h = (nodes / lengths).unsqueeze(1)
        elif self.pooling_method == 'attention':
            nodes = torch.cat(nodes, dim=1)
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask_expand = att_mask.float().unsqueeze(2).expand_as(nodes)
            att_weights = basic.masked_softmax(
                logits=self.combiner(nodes), mask=att_mask)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            h = (att_weights_expand * att_mask_expand * nodes).sum(1).unsqueeze(1)
        else:
            assert self.pooling_method is None
        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1)

    @staticmethod
    def compose(old_state, new_state, tree_mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        select_mask_expand = tree_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = tree_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, selected_h

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

