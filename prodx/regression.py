
from anndata import AnnData
from jax.experimental.sparse import BCOO
from numpyro.infer import SVI, Trace_ELBO
from patsy import dmatrix
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import re

# X: [ncells, ngenes]
# A: [ncells, ngenes] (sparse matrix)
# design: [ncells, ncovariates]
# negctrl_mask: [ncovariates, ngenes]
# batch_size: int
def model(X: jax.Array, A: BCOO, design: jax.Array, confounder_design: Optional[jax.Array], negctrl_mask: jax.Array, batch_size: int):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w = numpyro.sample("w", dist.Normal(jnp.zeros((ncovariates, ngenes)), jnp.full((ncovariates, ngenes), 1.0)))
    c = numpyro.sample("c", dist.LogNormal(jnp.zeros(ngenes), jnp.full(ngenes, 1.0)))

    if confounder_design is not None:
        nconfounders = confounder_design.shape[1]
        b_mul = numpyro.sample("b_mul", dist.Normal(jnp.zeros((nconfounders, 1)), jnp.full((nconfounders, 1), 1.0)))
        b_add = numpyro.sample("b_add", dist.LogNormal(jnp.zeros((nconfounders, 1)), jnp.full((nconfounders, 1), 1.0)))

    with numpyro.plate("cells", ncells, subsample_size=batch_size, dim=-2) as ind, numpyro.plate("genes", ngenes, dim=-1):
        X_batch = X[ind] # [batch_size, ngenes]
        design_batch = design[ind] # [batch_size, ncovariates]

        base_exp = design_batch@(w*negctrl_mask)
        if confounder_design is not None:
            confounder_design_batch = confounder_design[ind] # [batch_size, nconfounders]
            nb_mean = jnp.exp(base_exp + confounder_design_batch@b_mul) + confounder_design_batch@b_add # [batch_size, ngenes]
        else:
            nb_mean = jnp.exp(base_exp) # [batch_size, ngenes]

        # TODO: We'll need to subsample the adjacency matrix once
        # we start trying to make this work again
        # if A is not None:
        #     # This actually is bit nutty. What I should be doing is passing the adjacency matrix here
        #     # and multiplying nb_mean to diffuse it
        #     b_diffusion = numpyro.sample(
        #         "b",
        #         dist.HalfCauchy(jnp.full(ngenes, 1e-1)))
        #     nb_mean += b_diffusion * (A @ nb_mean)

        numpyro.sample(
            "X",
            dist.NegativeBinomial2(
                mean=nb_mean,
                concentration=jnp.repeat(jnp.expand_dims(c, 0), nb_mean.shape[0], axis=0),
            ),
            obs=X_batch)

def guide(X, A, design, confounder_design: Optional[jax.Array], negctrl_mask, batch_size):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w_mu_q = numpyro.param("w_mu_q", jnp.zeros((ncovariates, ngenes)))
    w_sigma_q = numpyro.param("w_sigma_q", jnp.ones((ncovariates, ngenes)), constraint=dist.constraints.positive)
    numpyro.sample("w", dist.Normal(w_mu_q, w_sigma_q))

    # if A is not None:
    #     b_mu_q = numpyro.param("b_mu_q", jnp.zeros(ngenes))
    #     b_sigma_q = numpyro.param("b_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
    #     numpyro.sample("b", dist.LogNormal(b_mu_q, b_sigma_q))

    c_mu_q = numpyro.param("c_mu_q", jnp.zeros(ngenes))
    c_sigma_q = numpyro.param("c_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
    numpyro.sample("c", dist.LogNormal(c_mu_q, c_sigma_q))

    if confounder_design is not None:
        nconfounders = confounder_design.shape[1]
        b_mul_mu_q = numpyro.param("b_mul_mu_q", jnp.zeros((nconfounders, 1)))
        b_mul_sigma_q = numpyro.param("b_mul_sigma_q", jnp.ones((nconfounders, 1)), constraint=dist.constraints.positive)
        numpyro.sample("b_mul", dist.Normal(b_mul_mu_q, b_mul_sigma_q))

        b_add_mu_q = numpyro.param("b_add_mu_q", jnp.zeros((nconfounders, 1)))
        b_add_sigma_q = numpyro.param("b_add_sigma_q", jnp.ones((nconfounders, 1)), constraint=dist.constraints.positive)
        numpyro.sample("b_add", dist.LogNormal(b_add_mu_q, b_add_sigma_q))

def run_inference_vi(X, A, design, confounder_design, negctrl_mask, batch_size, nsamples):
    optimizer = numpyro.optim.Adam(step_size=0.02)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(0)
    (params, _, _) = svi.run(
        key, nsamples,
        X=jnp.array(X, dtype=jnp.float32),
        A=A,
        design=jnp.array(design, dtype=jnp.float32),
        confounder_design=jnp.array(confounder_design, dtype=jnp.float32),
        negctrl_mask=jnp.array(negctrl_mask, dtype=jnp.float32),
        batch_size=batch_size)
    params = {
        k: np.asarray(v) for k, v in params.items()
    }
    return params

class DEModel:
    def __init__(self, adata: AnnData, formula: str, confounder_formula: Optional[str]=None, negctrl_pat:str="^NegControl", model_segmentation_error: bool=False, max_edge_dist=15.0):
        self.adata = adata
        self.design = dmatrix(formula, adata.obs)
        if confounder_formula is not None:
            self.confounder_design = dmatrix(confounder_formula, adata.obs)
        else:
            self.confounder_design = None

        ncovariates = self.design.shape[1]

        negctrl_mask = np.array([re.match(negctrl_pat, gene) is None for gene in adata.var_names], dtype=bool)
        nnegctrls = np.sum(~negctrl_mask)
        negctrl_mask = np.repeat(np.expand_dims(negctrl_mask, 0), ncovariates, axis=0) # mask out regression coefficients for negative controls

        # TODO: I don't think this masking is really necessary, but maybe it's safer to do so.
        negctrl_mask[0,:] = True # allow intercepts for negative controls
        # negctrl_mask[:] = True # allow intercepts for negative controls
        self.negctrl_mask = negctrl_mask
        print(f"{nnegctrls} negative controls")

        if model_segmentation_error:
            if "spatial" not in adata.obsm:
                raise ValueError("Spatial information is required for segmentation error modeling.")

            xys = np.asarray(adata.obsm["spatial"])
            tri = Delaunay(xys)
            tri_indptr, tri_indices = tri.vertex_neighbor_vertices

            edge_from = []
            edge_to = []
            for i in range(adata.shape[0]):
                for j in tri_indices[tri_indptr[i]:tri_indptr[i+1]]:
                    if i != j:
                        edge_from.append(i)
                        edge_to.append(j)

            edge_from = np.array(edge_from, dtype=np.int32)
            edge_to = np.array(edge_to, dtype=np.int32)

            distances = np.linalg.norm(xys[edge_from] - xys[edge_to], axis=1)
            distance_mask = distances <= max_edge_dist

            edge_from = edge_from[distance_mask]
            edge_to = edge_to[distance_mask]

            edges = jnp.stack((edge_from, edge_to), axis=1)
            self.A = BCOO((jnp.ones(len(edge_from)), edges), shape=(adata.shape[0], adata.shape[0]))
        else:
            self.A = None

    @property
    def design_matrix(self):
        return self.design

    @property
    def confounder_matrix(self):
        return self.confounder_design

    def fit(self, nsamples=6000, batch_size=1024):
        # TODO: We should try doing proper data loading and train on GPU.
        numpyro.set_platform("cpu")

        params = run_inference_vi(
            self.adata.X,
            self.A,
            self.design,
            self.confounder_design,
            self.negctrl_mask,
            batch_size=batch_size,
            nsamples=nsamples)

        # posterior mean coefficient estimates (should we convert these to log2?)
        coefs = pd.DataFrame(
            np.asarray(params["w_mu_q"]).transpose(),
            columns=self.design.design_info.column_names,
            index=self.adata.var_names,
        )

        sds = pd.DataFrame(
            np.asarray(params["w_sigma_q"]).transpose(),
            columns=self.design.design_info.column_names,
            index=self.adata.var_names,
        )

        return coefs, sds
