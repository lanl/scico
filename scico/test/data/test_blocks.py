from types import SimpleNamespace

import numpy as np

import imageio.v3 as iio
import pytest

from scico.data import _blocks as db


def test_split_list_deterministic_preserves_order() -> None:
    items = ["a", "b", "c", "d", "e", "f"]
    parts = db.split_list(items, [0.5, 0.5], randomize=False)
    assert parts == [["a", "b", "c"], ["d", "e", "f"]]


def test_split_list_randomized_preserves_relative_order_and_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [10, 20, 30, 40, 50, 60]
    # Force a deterministic shuffle order via monkeypatching random.shuffle.
    monkeypatch.setattr("random.shuffle", lambda seq: seq.reverse())

    parts = db.split_list(items, [1 / 3, 2 / 3], randomize=True)
    flattened = [elem for part in parts for elem in part]
    assert sorted(flattened) == items
    # Each partition's indices are sorted, so elements within a partition remain in original order.
    assert parts[0] == sorted(parts[0])
    assert parts[1] == sorted(parts[1])


def test_split_list_rescales_fractions() -> None:
    # Fractions that do not sum to 1.0 are rescaled rather than rejected, so
    # [0.2, 0.3] partitions in the same way as [0.4, 0.6].
    items = [1, 2, 3, 4, 5]
    parts = db.split_list(items, [0.2, 0.3], randomize=False)
    assert parts == [[1, 2], [3, 4, 5]]
    assert parts == db.split_list(items, [0.4, 0.6], randomize=False)


def test_image_size_and_channels() -> None:
    props_2d = SimpleNamespace(shape=(16, 32), dtype=np.uint8)
    assert db.image_size(props_2d) == 16 * 32 * 1
    assert db.image_channels(props_2d) == 1

    props_3d = SimpleNamespace(shape=(10, 20, 4), dtype=np.float32)
    assert db.image_size(props_3d) == 10 * 20 * 4 * 4
    assert db.image_channels(props_3d) == 4


def test_extract_blocks_2d_and_tuple_input() -> None:
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    blocks = db.extract_blocks(img, block_shape=(2, 2), block_step=(2, 2))
    assert blocks.shape == (2, 2, 2, 2)
    assert np.array_equal(blocks[0, 0], np.array([[0, 1], [4, 5]], dtype=np.uint8))
    assert np.array_equal(blocks[1, 1], np.array([[10, 11], [14, 15]], dtype=np.uint8))

    # Tuple of 2D arrays stacks along the last axis into a 3D image.
    ch0 = np.zeros((4, 4), dtype=np.uint8)
    ch1 = np.ones((4, 4), dtype=np.uint8)
    stacked_blocks = db.extract_blocks((ch0, ch1), block_shape=(2, 2), block_step=(2, 2))
    assert stacked_blocks.shape == (2, 2, 2, 2, 2)
    assert np.array_equal(stacked_blocks[0, 0, :, :, 0], np.zeros((2, 2), dtype=np.uint8))
    assert np.array_equal(stacked_blocks[0, 0, :, :, 1], np.ones((2, 2), dtype=np.uint8))


def test_extract_blocks_3d_block_on_3d_array() -> None:
    """Test 3D block extraction on a 3D array with no trailing dimensions."""
    volume = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
    blocks = db.extract_blocks(volume, block_shape=(2, 2, 2), block_step=(1, 1, 1))

    # Along each axis of length 3 with block_size=2 and step=1, there are 2 blocks -> (2, 2, 2, 2, 2, 2).
    assert blocks.shape == (2, 2, 2, 2, 2, 2)
    assert blocks.dtype == np.int32
    assert np.array_equal(blocks[0, 0, 0], volume[0:2, 0:2, 0:2])
    assert np.array_equal(blocks[1, 1, 1], volume[1:3, 1:3, 1:3])


def test_extract_blocks_3d_block_on_4d_array_preserves_trailing_dim() -> None:
    """Test 3D block extraction on a 4D array where axis 3 is a trailing channel dimension."""
    stack = np.arange(4 * 4 * 4 * 2, dtype=np.float32).reshape(4, 4, 4, 2)
    blocks = db.extract_blocks(stack, block_shape=(2, 2, 2), block_step=(2, 2, 2))

    # 2x2x2 block grid, 2x2x2 block voxels, and 2 trailing channels per voxel.
    assert blocks.shape == (2, 2, 2, 2, 2, 2, 2)
    assert blocks.dtype == np.float32
    assert np.array_equal(blocks[0, 0, 0], stack[0:2, 0:2, 0:2, :])
    assert np.array_equal(blocks[1, 0, 1], stack[2:4, 0:2, 2:4, :])


def test_extract_blocks_default_step_on_3d_array() -> None:
    """Test that omitting block_step defaults to a step of 1 along each block axis."""
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    blocks = db.extract_blocks(volume, block_shape=(2, 2, 2))

    # Axis 0: (2-2)/1 + 1 = 1; Axis 1: (3-2)/1 + 1 = 2; Axis 2: (4-2)/1 + 1 = 3.
    assert blocks.shape == (1, 2, 3, 2, 2, 2)
    assert blocks.dtype == np.uint16
    assert np.array_equal(blocks[0, 1, 2], volume[0:2, 1:3, 2:4])


def test_num_image_blocks() -> None:
    # (8 - 4) / 2 + 1 = 3 along each axis -> 3 * 3 = 9 blocks.
    assert db.num_image_blocks((8, 8), (4, 4), (2, 2)) == 9


def test_select_image_files_and_get_image_numblocks(tmp_path) -> None:
    img_a = np.zeros((8, 8, 3), dtype=np.uint8)
    img_b = np.ones((8, 8, 3), dtype=np.uint8)
    img_gray = np.zeros((8, 8), dtype=np.uint8)

    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    path_gray = tmp_path / "gray.png"

    iio.imwrite(path_a, img_a)
    iio.imwrite(path_b, img_b)
    iio.imwrite(path_gray, img_gray)

    selected = db.select_image_files(str(tmp_path / "*.png"), num_channels=3, dtype=np.uint8)
    assert selected == [str(path_a), str(path_b)]

    counts = db.get_image_numblocks(selected, block_shape=(4, 4), block_step=(4, 4))
    assert counts.dtype == np.uint32
    assert np.array_equal(counts, np.array([4, 4], dtype=np.uint32))


def test_get_image_blocks_samples_requested_count_and_validates_limit(tmp_path) -> None:
    img_a = np.full((4, 4, 3), 10, dtype=np.uint8)
    img_b = np.full((4, 4, 3), 20, dtype=np.uint8)

    path_a = tmp_path / "img1.png"
    path_b = tmp_path / "img2.png"
    iio.imwrite(path_a, img_a)
    iio.imwrite(path_b, img_b)

    paths = [str(path_a), str(path_b)]
    # Each 4x4 image with block_shape=(2, 2) and block_step=(2, 2) yields 4 blocks -> 8 total.
    sampled = db.get_image_blocks(paths, block_shape=(2, 2), block_step=(2, 2), num_blocks=5)
    assert sampled.shape == (5, 2, 2, 3)
    assert sampled.dtype == np.uint8

    with pytest.raises(ValueError, match="8 available and 9 requested"):
        db.get_image_blocks(paths, block_shape=(2, 2), block_step=(2, 2), num_blocks=9)


def test_get_image_blocks_zero_requested_returns_empty_array(tmp_path) -> None:
    """Test that requesting num_blocks == 0 returns an empty array without error."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    path = tmp_path / "single.png"
    iio.imwrite(path, img)

    empty_blocks = db.get_image_blocks(
        [str(path)],
        block_shape=(2, 2),
        block_step=(2, 2),
        num_blocks=0,
    )
    assert empty_blocks.size == 0
    assert empty_blocks.ndim >= 1
