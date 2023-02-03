import torch
import torch.nn as nn


def knn(query, data, k=10):

    assert data.shape[1] == query.shape[1]

    M = torch.cdist(query, data)
    # M = 1 - torch.mm(query, data.t())
    v, ind = M.topk(k, largest=False)

    return v, ind[:, 0:min(k, data.shape[0])].to(torch.long)


def sample_knn_labels(query_embd, y_query, prior_embd, labels, k=10, n_class=10, weighted=False):

    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), torch.randint(0, k+1, (n_sample,))]

    # convert labels to bincount (row wise)
    y_one_hot_batch = nn.functional.one_hot(neighbour_label_distribution, num_classes=n_class).float()

    # max_agree, _ = torch.max(torch.sum(y_one_hot_batch, dim=1), dim=1)

    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]

    # normalize max count as weight
    if weighted:
        weights = neighbour_freq / torch.sum(neighbour_freq)
    else:
        weights = 1/ n_sample * torch.ones([n_sample]).to(query_embd.device)

    return sampled_labels, torch.squeeze(weights)


if __name__ == "__main__":

    n_class = 10
    n_test_sample = 6
    n_train_sample = 100
    query_embd = torch.rand([n_test_sample, n_class])
    # query_embd = torch.softmax(100 * query_embd, dim=0)
    prior_embd = torch.rand([n_train_sample, n_class])
    # prior_embd = torch.softmax(100 * prior_embd, dim=0)
    labels = torch.argmax(prior_embd, dim=1)
    y_query = torch.randint(0, n_class, [n_test_sample])

    s, w = sample_knn_labels(query_embd, y_query, prior_embd, labels, k=10, weighted=True)

    print(s, w)







