import pytest

from flax import nnx
from scico.flax_nnx.autoencoders.autoencoders import (
    MLPDecoder,
    MLPEncoder,
)
from scico.flax_nnx.autoencoders.varautoencoders import (
    VAE,
    Conditioner,
    ConvVarAutoEncoder,
    MLPVarAutoEncoder,
    VarEncoder,
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
        self.shape_pre_latent = (4, 4, 4)

        self.mlpencoder1 = MLPEncoder(
            dim_in=self.N,
            widths_encoder=self.widths_encoder,
            dim_latent=self.dim_latent,
            rngs=nnx.Rngs(10),
        )

        self.mlpencoder2 = MLPEncoder(
            dim_in=self.N,
            widths_encoder=(10, 10),
            dim_latent=self.dim_latent,
            rngs=nnx.Rngs(100),
        )

        self.mlpdecoder = MLPDecoder(
            dim_latent=self.dim_latent,
            widths_decoder=self.widths_decoder,
            shape_out=(self.N, self.N, self.chn),
            rngs=nnx.Rngs(1000),
        )

    def test_varencoder_default(self):
        try:
            varencoder = VarEncoder(mean_block=self.mlpencoder1, logvar_block=self.mlpencoder2)
        except Exception as e:
            print(e)
            assert 0

    def test_conditioner_default(self):
        try:
            conditionner = Conditioner(
                processing_block=self.mlpencoder1,
                conditioning_block=self.mlpencoder2,
            )
        except Exception as e:
            print(e)
            assert 0

    def test_vae_default(self):
        try:
            vae = VAE(
                encoder=self.mlpencoder1,
                decoder=self.mlpdecoder,
            )
        except Exception as e:
            print(e)
            assert 0

    def test_vae_cond(self):
        varencoder = VarEncoder(mean_block=self.mlpencoder1, logvar_block=self.mlpencoder2)
        conditioner = Conditioner(
            processing_block=self.mlpencoder1, conditioning_block=self.mlpencoder2
        )
        try:
            vae = VAE(
                encoder=varencoder,
                decoder=self.mlpdecoder,
                conditioner=conditioner,
            )
        except Exception as e:
            print(e)
            assert 0

    def test_mlpvae_default(self):
        try:
            vae = MLPVarAutoEncoder(
                dim_in=self.N,
                widths_mean_encoder=self.widths_encoder,
                widths_logvar_encoder=self.widths_encoder,
                dim_latent=self.dim_latent,
                widths_decoder=self.widths_encoder,
                shape_out=(self.N, self.N, self.chn),
                rngs=nnx.Rngs(4),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_mlpvae_conditioning(self):
        try:
            vae = MLPVarAutoEncoder(
                dim_in=self.N,
                widths_mean_encoder=self.widths_encoder,
                widths_logvar_encoder=self.widths_encoder,
                dim_latent=self.dim_latent,
                widths_decoder=self.widths_encoder,
                shape_out=(self.N, self.N, self.chn),
                conditional=True,
                dim_cond_in=self.N,
                widths_condproc_encoder=self.widths_encoder,
                widths_cond_encoder=self.widths_encoder,
                rngs=nnx.Rngs(44),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_mlpvae_conditioning_except(self):
        with pytest.raises(AssertionError):
            vae = MLPVarAutoEncoder(
                dim_in=self.N,
                widths_mean_encoder=self.widths_encoder,
                widths_logvar_encoder=self.widths_encoder,
                dim_latent=self.dim_latent,
                widths_decoder=self.widths_encoder,
                shape_out=(self.N, self.N, self.chn),
                conditional=True,
                rngs=nnx.Rngs(444),
            )

    def test_convvae_default(self):
        try:
            vae = ConvVarAutoEncoder(
                shape_in=(self.N, self.N),
                channels=self.chn,
                filters_mean_encoder=self.filters_encoder,
                filters_logvar_encoder=self.filters_encoder,
                dim_latent=self.dim_latent,
                filters_decoder=self.filters_decoder,
                rngs=nnx.Rngs(4),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_convvae_conditioning(self):
        try:
            vae = ConvVarAutoEncoder(
                shape_in=(self.N, self.N),
                channels=self.chn,
                filters_mean_encoder=self.filters_encoder,
                filters_logvar_encoder=self.filters_encoder,
                dim_latent=self.dim_latent,
                filters_decoder=self.filters_decoder,
                conditional=True,
                widths_condproc_encoder=self.widths_encoder,
                shape_cond_in=(self.N, self.N),
                channels_cond=self.chn,
                filters_cond_encoder=self.filters_encoder,
                rngs=nnx.Rngs(44),
            )
        except Exception as e:
            print(e)
            assert 0

    def test_convvae_conditioning_except(self):
        with pytest.raises(AssertionError):
            vae = ConvVarAutoEncoder(
                shape_in=(self.N, self.N),
                channels=self.chn,
                filters_mean_encoder=self.filters_encoder,
                filters_logvar_encoder=self.filters_encoder,
                dim_latent=self.dim_latent,
                filters_decoder=self.filters_decoder,
                conditional=True,
                rngs=nnx.Rngs(444),
            )
