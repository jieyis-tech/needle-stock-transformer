import unittest
import numpy as np
import torch
import torch.nn.functional as F

import master_needle
import needle as ndl
from master_needle.attention import attention, dense_attention, dense_vjp_attention
from master_needle.model import MASTER, named_parameters

METRICS = {'attention_max_abs': 0.0, 'attention_gradient_max_abs': 0.0,
           'master_max_abs': 0.0, 'master_parameter_gradient_max_abs': 0.0}


def torch_master(x, market, model):
    """Independent functional reference. PyTorch is used ONLY for validation."""
    p = {k: torch.tensor(v.numpy(), requires_grad=True) for k, v in named_parameters(model).items()}
    def linear(z, key):
        h = z @ p[key + '.weight']
        return h + p[key + '.bias'] if key + '.bias' in p else h
    def norm(z, key):
        return F.layer_norm(z, (model.dim,), p[key + '.weight'], p[key + '.bias'], 1e-5)
    def block(z, key, heads, spatial):
        z = norm(z, key + '.norm1')
        seq = z.transpose(0, 1) if spatial else z
        b, n, d = seq.shape
        q, k, v = [linear(seq, key + '.' + name).reshape(b, n, heads, d // heads).transpose(1, 2) for name in ('qtrans','ktrans','vtrans')]
        scale = (d // heads) ** -0.5 if spatial else 1.0
        h = torch.softmax((q @ k.transpose(-1,-2)) * scale, -1) @ v
        h = h.transpose(1, 2).reshape(b, n, d)
        if spatial:
            h = h.transpose(0, 1)
        h = norm(z + h, key + '.norm2')
        return h + linear(F.relu(linear(h, key + '.ff1')), key + '.ff2')
    gate = torch.softmax(linear(market, 'gate') / model.beta, -1) * model.input_dim
    h = linear(x * gate, 'embed') + torch.tensor(model.position.numpy()[:x.shape[1]])
    h = block(h, 'temporal', model.temporal.heads, False)
    h = block(h, 'spatial', model.spatial.heads, True)
    z = linear(h, 'pool')
    a = torch.softmax((z @ z[:, -1, :, None]).squeeze(-1), 1)
    return linear((h * a[..., None]).sum(1), 'output').squeeze(-1), p


class AttentionTests(unittest.TestCase):
    def check_case(self, shape, nk, dv, block, dtype, scale=None):
        rng = np.random.default_rng(28)
        arrays = [rng.normal(size=shape).astype(dtype),
                  rng.normal(size=shape[:-2] + (nk, shape[-1])).astype(dtype),
                  rng.normal(size=shape[:-2] + (nk, dv)).astype(dtype)]
        grad = rng.normal(size=shape[:-1] + (dv,)).astype(dtype)
        nodes = [ndl.Tensor(a) for a in arrays]
        result = attention(*nodes, block_size=block, scale=scale)
        result.backward(ndl.Tensor(grad, requires_grad=False))
        refs = [torch.tensor(a, requires_grad=True) for a in arrays]
        s = shape[-1] ** -0.5 if scale is None else scale
        expected = torch.softmax((refs[0] @ refs[1].transpose(-1,-2)) * s, -1) @ refs[2]
        expected.backward(torch.tensor(grad))
        METRICS['attention_max_abs'] = max(METRICS['attention_max_abs'],float(np.max(np.abs(result.numpy()-expected.detach().numpy()))))
        tol = 4e-5 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(result.numpy(), expected.detach().numpy(), rtol=tol, atol=tol)
        for node, ref in zip(nodes, refs):
            METRICS['attention_gradient_max_abs'] = max(METRICS['attention_gradient_max_abs'],float(np.max(np.abs(node.grad.numpy()-ref.grad.numpy()))))
            np.testing.assert_allclose(node.grad.numpy(), ref.grad.numpy(), rtol=tol, atol=tol)
        self.assertLessEqual(result.op.max_score_elements, block * block)
        self.assertEqual(result.op.lse.size, np.prod(shape[:-1]))

    def test_shapes_dtypes_and_partial_tiles(self):
        for dtype in (np.float32, np.float64):
            for shape, nk, dv, block in [((7, 4), 9, 3, 3), ((2,3,7,4), 5, 6, 4), ((1,1), 1, 1, 8)]:
                with self.subTest(dtype=dtype, shape=shape):
                    self.check_case(shape,nk,dv,block,dtype)

    def test_temporal_unscaled(self):
        self.check_case((2,4,5,3),5,3,2,np.float64,scale=1.0)

    def test_dense_first_order_control(self):
        rng = np.random.default_rng(73)
        arrays = [rng.normal(size=(2,3,7,4)),rng.normal(size=(2,3,9,4)),rng.normal(size=(2,3,9,5))]
        groups = [[ndl.Tensor(a) for a in arrays] for _ in range(2)]
        grad = ndl.Tensor(rng.normal(size=(2,3,7,5)),requires_grad=False)
        outputs = [dense_vjp_attention(*groups[0]),attention(*groups[1],block_size=3)]
        for out in outputs:
            out.backward(grad)
        np.testing.assert_allclose(outputs[0].numpy(),outputs[1].numpy(),atol=1e-12)
        for a,b in zip(*groups):
            np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),atol=1e-12)

    def test_finite_difference(self):
        rng = np.random.default_rng(6)
        arrays = [rng.normal(size=(4,3)) for _ in range(3)]
        nodes = [ndl.Tensor(a) for a in arrays]
        weight = rng.normal(size=(4,3))
        attention(*nodes,block_size=2).backward(ndl.Tensor(weight,requires_grad=False))
        eps = 1e-5
        for a, node in zip(arrays,nodes):
            for idx in np.ndindex(a.shape):
                old = a[idx]
                a[idx] = old + eps
                plus = (attention(*[ndl.Tensor(x) for x in arrays],block_size=2).numpy()*weight).sum()
                a[idx] = old - eps
                minus = (attention(*[ndl.Tensor(x) for x in arrays],block_size=2).numpy()*weight).sum()
                a[idx] = old
                self.assertAlmostEqual((plus-minus)/(2*eps),node.grad.numpy()[idx],places=7)

    def test_stability_and_reject_invalid(self):
        self.check_case((2,7,4),9,3,2,np.float64,scale=50.0)
        q = ndl.Tensor(np.zeros((2,3),dtype='float32'))
        with self.assertRaises(ValueError): attention(q,q,q,block_size=0)
        bad = ndl.Tensor(np.full((2,3),np.nan,dtype='float32'))
        with self.assertRaises(ValueError): attention(bad,q,q)

    def test_gradient_accumulates_through_shared_inputs(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(5,4))
        x = ndl.Tensor(a)
        attention(x,x,x,block_size=2).sum().backward()
        ref = torch.tensor(a,requires_grad=True)
        (torch.softmax(ref@ref.T / 2,-1)@ref).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(),ref.grad.numpy(),atol=1e-10)


class MasterTests(unittest.TestCase):
    def test_full_model_forward_and_all_parameter_gradients(self):
        for mode in ('dense','tiled'):
            with self.subTest(mode=mode):
                np.random.seed(5)
                model = MASTER(6,market_dim=4,dim=8,t_heads=2,s_heads=2,mode=mode,block_size=3,dtype='float64')
                rng = np.random.default_rng(5)
                xa, ma = rng.normal(size=(7,5,6)), rng.normal(size=4)
                x, m = ndl.Tensor(xa), ndl.Tensor(ma)
                tx, tm = torch.tensor(xa,requires_grad=True), torch.tensor(ma,requires_grad=True)
                expected, refs = torch_master(tx,tm,model)
                out = model(x,m)
                out.sum().backward()
                expected.sum().backward()
                METRICS['master_max_abs'] = max(METRICS['master_max_abs'],float(np.max(np.abs(out.numpy()-expected.detach().numpy()))))
                np.testing.assert_allclose(out.numpy(),expected.detach().numpy(),atol=1e-8,rtol=1e-8)
                np.testing.assert_allclose(x.grad.numpy(),tx.grad.numpy(),atol=1e-8,rtol=1e-8)
                np.testing.assert_allclose(m.grad.numpy(),tm.grad.numpy(),atol=1e-8,rtol=1e-8)
                self.assertEqual(len(refs),len(model.parameters()))
                for name,p in named_parameters(model).items():
                    METRICS['master_parameter_gradient_max_abs'] = max(METRICS['master_parameter_gradient_max_abs'],float(np.max(np.abs(p.grad.numpy()-refs[name].grad.numpy()))))
                    np.testing.assert_allclose(p.grad.numpy(),refs[name].grad.numpy(),atol=1e-8,rtol=1e-8,err_msg=name)

    def test_training_dense_tiled_parameter_trajectory(self):
        models = []
        for mode in ('dense','tiled'):
            np.random.seed(18)
            models.append(MASTER(6,market_dim=4,dim=8,t_heads=2,mode=mode,block_size=3))
        opts = [ndl.optim.Adam(m.parameters(),lr=.001) for m in models]
        rng = np.random.default_rng(18)
        for _ in range(5):
            xa, ma, ya = rng.normal(size=(7,5,6)).astype('float32'),rng.normal(size=4).astype('float32'),rng.normal(size=7).astype('float32')
            for model,opt in zip(models,opts):
                opt.reset_grad()
                delta = model(ndl.Tensor(xa),ndl.Tensor(ma)) - ndl.Tensor(ya)
                ((delta*delta).sum()/7).backward()
                opt.step()
            for a,b in zip(models[0].parameters(),models[1].parameters()):
                np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-5,rtol=1e-5)

    def test_stock_permutation_equivariance(self):
        np.random.seed(2)
        model = MASTER(6,market_dim=4,dim=8,t_heads=2,block_size=3)
        rng = np.random.default_rng(2)
        x = rng.normal(size=(7,4,6)).astype('float32')
        m = ndl.Tensor(rng.normal(size=4).astype('float32'))
        perm = rng.permutation(7)
        a, b = model(ndl.Tensor(x),m).numpy(),model(ndl.Tensor(x[perm]),m).numpy()
        np.testing.assert_allclose(a[perm],b,atol=3e-5,rtol=3e-5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

