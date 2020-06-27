import numpy as np
import editDistance as ed
import heapq as hq


def ctc_loss(params, seq, blank=0, is_prob=True):
    """
    CTC loss function.
    params - n x m 的矩阵 n-D 概率分布，m是m步长 
    seq - 输入序列.
    is_prob - 参数是否传过softmax
    
    """
    seq_len = seq.shape[0]  # label的长度
    num_phones = params.shape[0]  # 类别数量
    L = 2 * seq_len + 1  # label增加了blank之后的长度
    T = params.shape[1]  # 输入时间序列长度

    alphas = np.zeros((L, T))
    betas = np.zeros((L, T))

    # 如果参数不是概率，转化为概率
    if not is_prob:
        params = params - np.max(params, axis=0)
        params = np.exp(params)
        params = params / np.sum(params, axis=0)

    # 初始化 alphas和前项推理参数
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    c = np.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / c
    llForward = np.log(c)
    for t in xrange(1, T):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in xrange(start, L):
            l = (s - 1) / 2
            # 为blank的情况
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            # 同样的label 两次
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        # 归一化当前T(防止下溢)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] = alphas[start:end, t] / c
        llForward += np.log(c)

    # 初始化betas和后向推理过程
    betas[-1, -1] = params[blank, -1]
    betas[-2, -1] = params[seq[-1], -1]
    c = np.sum(betas[:, -1])
    betas[:, -1] = betas[:, -1] / c
    llBackward = np.log(c)
    for t in xrange(T - 2, -1, -1):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in xrange(end - 1, -1, -1):
            l = (s - 1) / 2
            # 空的情况
            if s % 2 == 0:
                if s == L - 1:
                    betas[s, t] = betas[s, t + 1] * params[blank, t]
                else:
                    betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[blank, t]
            # 同样的label两次
            elif s == L - 2 or seq[l] == seq[l + 1]:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
            else:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) \
                              * params[seq[l], t]

        c = np.sum(betas[start:end, t])
        betas[start:end, t] = betas[start:end, t] / c
        llBackward += np.log(c)

    # 计算梯度 
    grad = np.zeros(params.shape)
    ab = alphas * betas
    for s in xrange(L):
        # blank
        if s % 2 == 0:
            grad[blank, :] += ab[s, :]
            ab[s, :] = ab[s, :] / params[blank, :]
        else:
            grad[seq[(s - 1) / 2], :] += ab[s, :]
            ab[s, :] = ab[s, :] / (params[seq[(s - 1) / 2], :])
    absum = np.sum(ab, axis=0)

    # 检查下溢
    llDiff = np.abs(llForward - llBackward)
    if llDiff > 1e-5 or np.sum(absum == 0) > 0:
        print("Diff in forward/backward LL : %f" % llDiff)
        print("Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0]))
        return -llForward, grad, True


grad = params - grad / (params * absum)

return -llForward, grad, False


def decode_best_path(probs, ref=None, blank=0):
    """
    解码最佳路径
    """

    # 计算最可能的path
    best_path = np.argmax(probs, axis=0).tolist()

    # 折叠phone 
    hyp = []
    for i, b in enumerate(best_path):
        # 忽略blank
        if b == blank:
            continue
        # 忽略重复label
        elif i != 0 and b == best_path[i - 1]:
            continue
        else:
            hyp.append(b)

    # 计算错误率
    dist = 0
    if ref is not None:
        ref = ref.tolist()
        dist, _, _, _, _ = ed.edit_distance(ref, hyp)

    return hyp, dist
