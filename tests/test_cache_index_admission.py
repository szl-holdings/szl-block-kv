# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""CPU admission regressions: invalid addressing has no cache side effects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

import pytest
import torch
import torch.nn.functional as F

from szl_block_kv import PagedCache, ReceiptChain, paged_attn, reshape_and_cache


@pytest.fixture
def write_case():
    cache = PagedCache(num_blocks=2, block_size=2, n_heads=1, d_head=2)
    cache.k.fill_(11)
    cache.v.fill_(29)
    k = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    return cache, k, k + 100


@pytest.mark.parametrize("slots", [
    pytest.param(torch.tensor([-1, 0]), id="negative_first"),
    pytest.param(torch.tensor([0, -1]), id="negative_last"),
    pytest.param(torch.tensor([-1, -1]), id="all_negative"),
    pytest.param(torch.tensor([4, 0]), id="upper_first"),
    pytest.param(torch.tensor([0, 4]), id="upper_last"),
    pytest.param(torch.tensor([0.0, 1.5]), id="fractional"),
    pytest.param(torch.tensor([0.0, 1.0]), id="integral_float"),
    pytest.param(torch.tensor([False, True]), id="bool"),
    pytest.param(torch.tensor([0.0, float("nan")]), id="nan"),
    pytest.param(torch.tensor([0.0, float("inf")]), id="infinity"),
    pytest.param(torch.tensor([0j, 1j]), id="complex"),
    pytest.param(torch.tensor([[0, 1]]), id="rank_two"),
    pytest.param(torch.tensor([0]), id="wrong_token_count"),
    pytest.param([0, 1], id="not_tensor"),
])
def test_invalid_write_slots_leave_both_caches_unchanged(write_case, slots):
    cache, k, v = write_case
    before_k, before_v = cache.k.clone(), cache.v.clone()
    with pytest.raises(ValueError):
        reshape_and_cache(k, v, cache, slots)
    assert torch.equal(cache.k, before_k)
    assert torch.equal(cache.v, before_v)


@pytest.mark.parametrize("which", ["k", "v"])
@pytest.mark.parametrize("shape", [(1, 1, 2), (2, 2, 2), (2, 1, 1), (2, 2)])
def test_bad_value_shape_preserves_both_caches(write_case, which, shape):
    cache, k, v = write_case
    if which == "k":
        k = torch.zeros(shape)
    else:
        v = torch.zeros(shape)
    before_k, before_v = cache.k.clone(), cache.v.clone()
    with pytest.raises(ValueError):
        reshape_and_cache(k, v, cache, torch.tensor([0, 1]))
    assert torch.equal(cache.k, before_k)
    assert torch.equal(cache.v, before_v)


@pytest.mark.parametrize("which", ["k", "v"])
@pytest.mark.parametrize("issue", ["dtype", "device", "layout"])
def test_bad_value_metadata_preserves_both_caches(write_case, which, issue):
    cache, k, v = write_case
    value = k if which == "k" else v
    if issue == "dtype":
        value = value.to(torch.float64)
    elif issue == "device":
        value = value.to("meta")  # Metadata-only mismatch; no accelerator allocation.
    else:
        value = value.to_sparse()
    if which == "k":
        k = value
    else:
        v = value
    before_k, before_v = cache.k.clone(), cache.v.clone()
    with pytest.raises(ValueError):
        reshape_and_cache(k, v, cache, torch.tensor([0, 1]))
    assert torch.equal(cache.k, before_k)
    assert torch.equal(cache.v, before_v)


@pytest.mark.parametrize("issue", ["shape", "dtype", "layout"])
def test_inconsistent_value_cache_preserves_both_caches(write_case, issue):
    cache, k, v = write_case
    if issue == "shape":
        cache.v = cache.v[:1]
    elif issue == "dtype":
        cache.v = cache.v.to(torch.float64)
    else:
        cache.v = cache.v.to_sparse()
    before_k = cache.k.clone()
    before_v = cache.v.to_dense().clone()
    with pytest.raises(ValueError):
        reshape_and_cache(k, v, cache, torch.tensor([0, 2]))
    assert torch.equal(cache.k, before_k)
    assert torch.equal(cache.v.to_dense(), before_v)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_matching_payload_and_cache_dtypes_preserve_ordinary_writes(dtype):
    cache = PagedCache(num_blocks=2, block_size=2, n_heads=1, d_head=2, dtype=dtype)
    k = torch.arange(4, dtype=dtype).reshape(2, 1, 2)
    v = k + 100
    reshape_and_cache(k, v, cache, torch.tensor([3, 0]))
    assert torch.equal(cache.k[1, 1], k[0])
    assert torch.equal(cache.k[0, 0], k[1])
    assert torch.equal(cache.v[1, 1], v[0])
    assert torch.equal(cache.v[0, 0], v[1])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_valid_integer_writes_keep_token_to_slot_mapping(write_case, dtype):
    cache, k, v = write_case
    expected_k, expected_v = cache.k.clone(), cache.v.clone()
    expected_k[1, 1], expected_k[0, 0] = k[0], k[1]
    expected_v[1, 1], expected_v[0, 0] = v[0], v[1]
    reshape_and_cache(k, v, cache, torch.tensor([3, 0], dtype=dtype))
    assert torch.equal(cache.k, expected_k)
    assert torch.equal(cache.v, expected_v)


class UnreadableCacheTensor:
    """Expose metadata but fail if admission touches any cache contents."""

    def __init__(self, tensor):
        self.shape, self.dtype, self.device = tensor.shape, tensor.dtype, tensor.device

    def __getitem__(self, _index):
        pytest.fail("cache contents read before all active pages were admitted")


@pytest.mark.parametrize("tables,contexts", [
    pytest.param([[0, -1], [-1, -1]], [2, 2], id="late_invalid_row"),
    pytest.param([[0, -1], [1, -1]], [2, 3], id="late_invalid_page"),
    pytest.param([[0, -1], [1, 2]], [2, 3], id="late_upper_page"),
])
def test_all_active_pages_are_admitted_before_any_cache_read(tables, contexts):
    cache = PagedCache(num_blocks=2, block_size=2, n_heads=1, d_head=2)
    cache.k, cache.v = UnreadableCacheTensor(cache.k), UnreadableCacheTensor(cache.v)
    chain = ReceiptChain()
    with pytest.raises(ValueError):
        paged_attn(torch.zeros(2, 1, 1, 2), cache, torch.tensor(tables),
                   torch.tensor(contexts), causal=False, chain=chain)
    assert len(chain) == 0


@pytest.mark.parametrize("tables,contexts", [
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([-1, 2]), id="negative_context"),
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([2, 5]), id="oversize_context"),
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([2.0, 2.5]), id="fractional_context"),
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([2.0, 2.0]), id="float_context"),
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([True, True]), id="bool_context"),
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([[2], [2]]), id="context_rank"),
    pytest.param(torch.tensor([[0, -1], [1, -1]]), torch.tensor([2]), id="context_batch"),
    pytest.param(torch.tensor([[0.5, -1], [1, -1]]), torch.tensor([2, 2]), id="fractional_page"),
    pytest.param(torch.tensor([[0.0, -1], [1, -1]]), torch.tensor([2, 2]), id="float_page"),
    pytest.param(torch.tensor([[False, False], [True, False]]), torch.tensor([2, 2]), id="bool_page"),
    pytest.param(torch.tensor([0, 1]), torch.tensor([2, 2]), id="table_rank"),
])
def test_invalid_read_admission_precedes_cache_reads_and_receipts(tables, contexts):
    cache = PagedCache(num_blocks=2, block_size=2, n_heads=1, d_head=2)
    cache.k, cache.v = UnreadableCacheTensor(cache.k), UnreadableCacheTensor(cache.v)
    chain = ReceiptChain()
    with pytest.raises(ValueError):
        paged_attn(torch.zeros(2, 1, 1, 2), cache, tables, contexts, causal=False, chain=chain)
    assert len(chain) == 0


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_inactive_minus_one_padding_matches_trimmed_sdpa(dtype):
    generator = torch.Generator().manual_seed(41)
    cache = PagedCache(num_blocks=2, block_size=2, n_heads=1, d_head=2)
    cache.k.copy_(torch.randn(cache.k.shape, generator=generator))
    cache.v.copy_(torch.randn(cache.v.shape, generator=generator))
    q = torch.randn(2, 1, 2, 2, generator=generator)
    chain = ReceiptChain()
    result = paged_attn(q, cache, torch.tensor([[1, 0, -1], [0, -1, -1]], dtype=dtype),
                        torch.tensor([3, 1], dtype=dtype), causal=False, chain=chain)
    expected = []
    for row, keys, values in (
        (0, torch.cat([cache.k[1], cache.k[0, :1]]), torch.cat([cache.v[1], cache.v[0, :1]])),
        (1, cache.k[0, :1], cache.v[0, :1]),
    ):
        expected.append(F.scaled_dot_product_attention(
            q[row:row + 1], keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=False))
    torch.testing.assert_close(result, torch.cat(expected), atol=1e-5, rtol=1e-5)
    assert chain.verify() == (True, 1, -1)
    assert chain._rows[0]["path"] == "torch_gather"
    assert chain._rows[0]["lambda"] == "Conjecture 1"


def test_zero_context_does_not_dereference_padding_or_fabricate_values():
    cache = PagedCache(num_blocks=1, block_size=2, n_heads=1, d_head=2)
    cache.v.fill_(99)
    result = paged_attn(torch.zeros(1, 1, 1, 2), cache, torch.tensor([[-1]]),
                        torch.tensor([0]), causal=False)
    assert torch.equal(result, torch.zeros_like(result))
