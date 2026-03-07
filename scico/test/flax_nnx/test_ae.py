import numpy as np

from flax import nnx
from scico.flax_nnx.autoencoders.autoencoders import (
    ConvAutoEncoder,
    ConvDecoder,
    ConvEncoder,
    MLPAutoEncoder,
    MLPDecoder,
    MLPEncoder,
)


class TestSet:
    def setup_method(self, setup_method):
        self.N = 128  # signal size
        self.chn = 1  # number of channels
        self.bsize = 16  # batch size

        # Dummy architecture configuration
        self.widths_encoder = (20, 10)
        self.dim_latent = 4
        self.widths_decoder = (10, 20)
        self.filters_encoder = (16, 8)
        self.filters_decoder = (8, 16)
        self.shape_latent = (4, 4, 4)

    def test_mlpencoder_default(self):
        try:
            mlpencoder = MLPEncoder(
                dim_in=self.N,
                widths_encoder=self.widths_encoder,
                dim_latent=self.dim_latent,
                rngs=nnx.Rngs(np.random.randint(10)),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_mlpdecoder_default(self):
        try:
            mlpdecoder = MLPDecoder(
                dim_latent=self.dim_latent,
                widths_decoder=self.widths_encoder,
                shape_out=(self.N, self.N, self.chn),
                rngs=nnx.Rngs(np.random.randint(10)),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_mlpae_default(self):
        try:
            mlpae = MLPAutoEncoder(
                dim_in=self.N,
                widths_encoder=self.widths_encoder,
                dim_latent=self.dim_latent,
                widths_decoder=self.widths_encoder,
                shape_out=(self.N, self.N, self.chn),
                rngs=nnx.Rngs(np.random.randint(10)),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_convencoder_default(self):
        try:
            convencoder = ConvEncoder(
                shape_in=(self.N, self.N),
                channels=self.chn,
                filters_encoder=self.filters_encoder,
                rngs=nnx.Rngs(np.random.randint(10)),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_convdecoder_default(self):
        try:
            convdecoder = ConvDecoder(
                filters_decoder=self.filters_decoder,
                shape_latent=self.shape_latent,
                channels=self.chn,
                rngs=nnx.Rngs(np.random.randint(10)),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_convae_default(self):
        try:
            convae = ConvAutoEncoder(
                shape_in=(self.N, self.N),
                channels=self.chn,
                filters_encoder=self.filters_encoder,
                filters_decoder=self.filters_decoder,
                rngs=nnx.Rngs(np.random.randint(10)),
            )
        except Exception as e:
            print(e)
            assert 0
