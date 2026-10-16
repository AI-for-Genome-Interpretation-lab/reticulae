"""reticulae: sparse convolutions in pure PyTorch. See README.md and docs/API.md."""
from reticulae.Convolutions import SparseTensor, SparseConvFn
from reticulae.Convolutions import SparseConv, SubMConv, SparseConvTranspose, SparseInverseConv
from reticulae.Convolutions import SparseConv1d, SparseConv2d, SparseConv3d, SubMConv1d, SubMConv2d, SubMConv3d
from reticulae.Convolutions import SparseConvTranspose1d, SparseConvTranspose2d, SparseConvTranspose3d
from reticulae.Convolutions import SparseInverseConv1d, SparseInverseConv2d, SparseInverseConv3d
from reticulae.Convolutions import FixedSparseConv, FixedSubMConv, FixedSparseConvTranspose, FixedSparseInverseConv
from reticulae.Pooling import SparsePool, SparseMaxPool, SparseAvgPool, FixedSparsePool, globalPool
from reticulae.Pooling import SparseMaxPool1d, SparseMaxPool2d, SparseMaxPool3d, SparseAvgPool1d, SparseAvgPool2d, SparseAvgPool3d
from reticulae.Storage import PackedArray, SparseImages
from reticulae.Hilbert import d2xy, hilbertXY, SparseHilbertCNN
