import torch
import torch.nn as nn


def log_sum_exp(x):
  """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
  """
  max_score = x.max(-1)[0]
  return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
  """General CRF module.
  The CRF module contain a inner Linear Layer which transform the input from features space to tag space.

  :param in_features: number of features for the input
  :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
  """

  def __init__(self, in_features, num_tags):
    super(CRF, self).__init__()

    self.num_tags = num_tags + 2
    self.start_idx = self.num_tags - 2
    self.stop_idx = self.num_tags - 1

    self.fc = nn.Linear(in_features, self.num_tags)

    # transition factor, Tij mean transition from j to i
    self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
    self.transitions.data[self.start_idx, :] = IMPOSSIBLE
    self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

  def forward(self, features, masks):
    """decode tags

    :param features: [B, L, C], batch of unary scores
    :param masks: [B, L] masks
    :return: (best_score, best_paths)
      best_score: [B]
      best_paths: [B, L]
    """
    features = self.fc(features)
    return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

  def loss(self, features, ys, masks):
    """negative log likelihood loss
    B: batch size, L: sequence length, D: dimension

    :param features: [B, L, D]
    :param ys: tags, [B, L]
    :param masks: masks for padding, [B, L]
    :return: loss
    """
    features = self.fc(features)

    L = features.size(1)
    masks_ = masks[:, :L].float()

    forward_score = self.__forward_algorithm(features, masks_)
    gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
    loss = (forward_score - gold_score).mean()
    return loss

  def __score_sentence(self, features, tags, masks):
    """Gives the score of a provided tag sequence

    :param features: [B, L, C]
    :param tags: [B, L]
    :param masks: [B, L]
    :return: [B] score in the log space
    """
    B, L, C = features.shape

    # emission score
    emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

    # transition score
    start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
    tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
    trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]] # [B, L]

    # last transition score to STOP tag
    last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
    last_score = self.transitions[self.stop_idx, last_tag]

    score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
    return score

  def __viterbi_decode(self, features, masks):
    """decode to tags using viterbi algorithm

    :param features: [B, L, C], batch of unary scores
    :param masks: [B, L] masks
    :return: (best_score, best_paths)
      best_score: [B]
      best_paths: [B, L]
    """
    B, L, C = features.shape

    bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

    # Initialize the viterbi variables in log space
    max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
    max_score[:, self.start_idx] = 0

    for t in range(L):
      mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
      emit_score_t = features[:, t]  # [B, C]

      # [B, 1, C] + [C, C]
      acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
      acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
      acc_score_t += emit_score_t
      max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

    # Transition to STOP_TAG
    max_score += self.transitions[self.stop_idx]
    best_score, best_tag = max_score.max(dim=-1)

    # Follow the back pointers to decode the best path.
    best_paths = []
    bps = bps.cpu().numpy()
    for b in range(B):
      best_tag_b = best_tag[b].item()
      seq_len = int(masks[b, :].sum().item())

      best_path = [best_tag_b]
      for bps_t in reversed(bps[b, :seq_len]):
        best_tag_b = bps_t[best_tag_b]
        best_path.append(best_tag_b)
      # drop the last tag and reverse the left
      best_paths.append(best_path[-2::-1])

    return best_score, best_paths

  def __forward_algorithm(self, features, masks):
    """calculate the partition function with forward algorithm.
    TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])

    :param features: features. [B, L, C]
    :param masks: [B, L] masks
    :return:  [B], score in the log space
    """
    B, L, C = features.shape

    scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
    scores[:, self.start_idx] = 0.
    trans = self.transitions.unsqueeze(0)  # [1, C, C]

    # Iterate through the sentence
    for t in range(L):
      emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
      score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
      score_t = log_sum_exp(score_t)  # [B, C]

      mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
      scores = score_t * mask_t + scores * (1 - mask_t)
    scores = log_sum_exp(scores + self.transitions[self.stop_idx])
    return scores
