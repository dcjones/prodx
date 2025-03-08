
from anndata import AnnData
from jax.experimental.sparse import BCOO
from numpyro.infer import SVI, Trace_ELBO
from pandas._libs.algos import nancorr
from patsy import dmatrix
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
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
def model(X: jax.Array, A: BCOO, design: jax.Array, negctrl_mask: jax.Array, batch_size: int):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w = numpyro.sample("w", dist.Normal(jnp.zeros((ncovariates, ngenes)), jnp.full((ncovariates, ngenes), 1.0)))
    b = numpyro.sample("b", dist.LogNormal(jnp.zeros((ncovariates-1, 1)), jnp.full((ncovariates-1, 1), 1.0)))
    c = numpyro.sample("c", dist.LogNormal(jnp.zeros(ngenes), jnp.full(ngenes, 1.0)))

    with numpyro.plate("cells", ncells, subsample_size=batch_size, dim=-2) as ind, numpyro.plate("genes", ngenes, dim=-1):
        X_batch = X[ind] # [batch_size, ngenes]
        design_batch = design[ind] # [batch_size, ncovariates]
        design_batch_no_intercept = design_batch[:,1:] # [batch_size, ncovariates-1]


        # TODO: Need to make the second term optional when there are no negative
        # controls to avoid degenerate solutions.


        # TODO: The bigger issue is regressing this additive noise term won't work for general design matrices.

        nb_mean = jnp.exp(design_batch@(w*negctrl_mask)) + design_batch_no_intercept@b # [batch_size, ngenes]

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

def guide(X, A, design, negctrl_mask, batch_size):
    ncells, ngenes = X.shape
    ncovariates = design.shape[1]

    w_mu_q = numpyro.param("w_mu_q", jnp.zeros((ncovariates, ngenes)))
    w_sigma_q = numpyro.param("w_sigma_q", jnp.ones((ncovariates, ngenes)), constraint=dist.constraints.positive)
    numpyro.sample("w", dist.Normal(w_mu_q, w_sigma_q))

    b_mu_q = numpyro.param("b_mu_q", jnp.zeros((ncovariates-1, 1)))
    b_sigma_q = numpyro.param("b_sigma_q", jnp.ones((ncovariates-1, 1)), constraint=dist.constraints.positive)
    numpyro.sample("b", dist.LogNormal(b_mu_q, b_sigma_q))

    # if A is not None:
    #     b_mu_q = numpyro.param("b_mu_q", jnp.zeros(ngenes))
    #     b_sigma_q = numpyro.param("b_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
    #     numpyro.sample("b", dist.LogNormal(b_mu_q, b_sigma_q))

    c_mu_q = numpyro.param("c_mu_q", jnp.zeros(ngenes))
    c_sigma_q = numpyro.param("c_sigma_q", jnp.ones(ngenes), constraint=dist.constraints.positive)
    numpyro.sample("c", dist.LogNormal(c_mu_q, c_sigma_q))


def run_inference_vi(X, A, design, negctrl_mask, batch_size, nsamples):
    optimizer = numpyro.optim.Adam(step_size=0.02)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(0)
    (params, _, _) = svi.run(
        key, nsamples,
        X=jnp.array(X, dtype=jnp.float32),
        A=A,
        design=jnp.array(design, dtype=jnp.float32),
        negctrl_mask=jnp.array(negctrl_mask, dtype=jnp.float32),
        batch_size=batch_size)
    params = {
        k: np.asarray(v) for k, v in params.items()
    }
    return params

class DEModel:
    def __init__(self, adata: AnnData, formula: str, negctrl_pat:str="^NegControl", model_segmentation_error: bool=False, max_edge_dist=15.0):
        self.adata = adata
        self.design = dmatrix(formula, adata.obs)
        ncovariates = self.design.shape[1]

        negctrl_mask = np.array([re.match(negctrl_pat, gene) is None for gene in adata.var_names], dtype=bool)
        nnegctrls = np.sum(~negctrl_mask)
        negctrl_mask = np.repeat(np.expand_dims(negctrl_mask, 0), ncovariates, axis=0) # mask out regression coefficients for negative controls
        negctrl_mask[0,:] = True # allow intercepts for negative controls
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

    def fit(self, nsamples=6000, batch_size=1024):
        # TODO: We should try doing proper data loading and train on GPU.
        numpyro.set_platform("cpu")

        params = run_inference_vi(
            self.adata.X,
            self.A,
            self.design,
            self.negctrl_mask,
            batch_size=batch_size,
            nsamples=nsamples)

        print(params["b_mu_q"])

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
