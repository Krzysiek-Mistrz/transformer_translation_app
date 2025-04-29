import argparse
import torch

from src.models.train import train_model
from src.data.inference import translate_sentence
from src.models.model import Seq2SeqTransformer
from data.data_loader import vocab_transform, BOS_IDX

def main():
    parser = argparse.ArgumentParser(description="DE→EN Transformer")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--epochs',    type=int, default=10)
    p_train.add_argument('--batch-size',type=int, default=128)

    p_trans = sub.add_parser('translate')
    p_trans.add_argument('sentence', type=str)
    p_trans.add_argument('--model',    type=str, default='transformer_de_to_en.pt')

    args = parser.parse_args()

    if args.cmd == 'train':
        train_model(epochs=args.epochs, batch_size=args.batch_size)
    elif args.cmd == 'translate':
        # rebuild architecture
        src_vocab = len(vocab_transform['de'])
        tgt_vocab = len(vocab_transform['en'])
        model = Seq2SeqTransformer(
            num_encoder_layers=3, num_decoder_layers=3,
            emb_size=512, nhead=8,
            src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab
        )
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        translation = translate_sentence(model, args.sentence)
        print("→", translation)

if __name__ == '__main__':
    main()