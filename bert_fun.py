from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging

def get_embeddings(document):
    # INPUT: string
    # OUTPUT: list of 12 tensors of shape 1 x token# x embedding_dim
    # logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = tokenizer.tokenize(document)
    print(len(tokenized_text))

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [] 
    segment_id = 0
    for i, token in enumerate(indexed_tokens):
        segment_ids.append(segment_id)
        if token == 1012: # token id for '.' 
            pass
            #segment_id += 1
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    '''
    # uncomment for CUDA:
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')
    '''

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    print([encoded_layer.shape for encoded_layer in encoded_layers])
    return encoded_layers

