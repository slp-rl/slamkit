from torch import FloatTensor, LongTensor
from torch.nn.functional import cross_entropy


def calc_nll(logits: FloatTensor, target: LongTensor, mask: LongTensor, len_norm: bool = True) -> FloatTensor:
    """
    calculate the negative log likelihood of the logits given the target
    :param logits: logits
    :param target: target
    :param mask: mask
    :param len_norm: whether to normalize the loss by the number of tokens
    :return: nll
    """
    # Calculate the cross-entropy loss for each sequence
    losses = cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        target.long().contiguous().view(-1), reduction='none')

    # Reshape the losses to match the original sequences
    losses = losses.view(*target.size())

    # Use the mask to ignore the losses of the padding tokens
    masked_losses = losses * mask

    # Sum the losses to get the total loss for each sequence
    ll = masked_losses.sum(dim=-1)
    if len_norm:
        return ll / mask.sum(dim=-1)
    return ll


def calc_ngram(text:str, nltk_word_tokenizer, n:int):
    tokens = nltk_word_tokenizer.tokenize(text)
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

def calc_auto_bleu(text, nltk_word_tokenizer, n):
    res = 0
    ngrams = calc_ngram(text, nltk_word_tokenizer, n)
    if len(ngrams) == 0:
        return 0
    for i in range(len(ngrams)):
        left = ngrams[:i]
        right = ngrams[i+1:]
        if ngrams[i] in left or ngrams[i] in right:
            res += 1
    return res/len(ngrams)