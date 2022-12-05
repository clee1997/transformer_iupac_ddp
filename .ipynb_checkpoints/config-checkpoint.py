import os

tok_ids = {
    'UNK_IDX' : 0, 
    'PAD_IDX' : 1, 
    'BOS_IDX' : 2, 
    'EOS_IDX' : 3
}


# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# csv_path = '/Users/chaeeunlee/Downloads/saved_3/merge_res/pair_df_merged.csv'
# saved_path = '/Users/chaeeunlee/Downloads/ckpt_saved/'
# saved_vocab_path = '/Users/chaeeunlee/Downloads/ckpt_saved/vocab_saved/vocab.pth'

csv_path = '/hdd1/chaeeun/transformer_iupac_dataset/pair_df_merged.csv'
saved_path = '/hdd1/chaeeun/transformer_iupac_dataset/ckpt_saved/'
saved_vocab_path = '/hdd1/chaeeun/transformer_iupac_dataset/ckpt_saved/vocab_saved/vocab.pth'

ckpt_path = os.path.join(saved_path, 'ckpt_epoch10.pt')

params = {
    'NUM_ENCODER_LAYERS' : 6,
    'NUM_DECODER_LAYERS' : 6,
    'EMB_SIZE' : 512,
    'NHEAD' : 8, # 8이었음. 
    'FFN_HID_DIM' : 512,
    'BATCH_SIZE' : 64, #128, ## 
    'NUM_EPOCHS' : 100
    # 'SRC_VOCAB_SIZE' : len(vocab['src']),
    # 'TGT_VOCAB_SIZE' : len(vocab['tgt'])
    }