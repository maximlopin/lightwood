# flake8: noqa
from lightwood.encoders.text.helpers.rnn_helpers import *
import logging
import math


class RnnEncoder:

    def __init__(self, encoded_vector_size=256):
        self._encoded_vector_size = encoded_vector_size
        self._encoder = None
        self._decoder = None
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False

    def prepare_encoder(self, window_size):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        hidden_size = self._encoded_vector_size
        self._encoder = EncoderRNN(window_size, hidden_size).to(device)
        self._decoder = DecoderRNN(hidden_size, window_size).to(device)

        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        ret = []
        with torch.no_grad():
            for row in column_data:
                row_data = torch.LongTensor(list(map(float, row.split())))
                encoder_hidden = self._encoder.initHidden()
                encoder_output, encoder_hidden = self._encoder(row_data, encoder_hidden)

                # input_tensor = tensorFromSentence(self._input_lang, row)
                # input_length = input_tensor.size(0)
                #
                # #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
                #
                # loss = 0
                #
                # for ei in range(input_length):
                #     encoder_output, encoder_hidden = self._encoder(
                #         input_tensor[ei], encoder_hidden)
                # encoder_outputs[ei] = encoder_output[0, 0]

                # use the last hidden state as the encoded vector
                ret += [encoder_hidden.tolist()[0][0]]

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor, max_length=100):

        ret = []
        with torch.no_grad():
            for decoder_hiddens in encoded_values_tensor:
                decoder_hidden = torch.FloatTensor([[decoder_hiddens.tolist()]])

                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

                decoded_output = []

                decoder_output, decoder_hidden = self._decoder(
                    decoder_input, decoder_hidden)

                # for di in range(max_length):
                #     decoder_output, decoder_hidden = self._decoder(
                #         decoder_input, decoder_hidden)
                #
                #     topv, topi = decoder_output.data.topk(1)
                #     if topi.item() == EOS_token:
                #         decoded_words.append('<EOS>')
                #         break
                #     else:
                #         decoded_words.append(self._output_lang.index2word[topi.item()])
                #
                #     decoder_input = topi.squeeze().detach()

                ret += [decoder_output]

        return ret


# only run the test if this file is called from debugger
if __name__ == "__main__":
    data = [" ".join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

    encoder = RnnEncoder(encoded_vector_size=10)
    encoder.prepare_encoder(10)
    encoder.encode(data)

    # test de decoder
    print(data)
    ret = encoder.encode(data)
    print('encoded vector:')
    print(ret)
    print('decoded vector')
    ret2 = encoder.decode(ret)
    print(ret2)
